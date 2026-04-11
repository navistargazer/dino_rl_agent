import torch
import torch.nn.functional as F
import numpy as np


def train_buffer(model, target_model, optimizer, batch, device, buffer_type, push_to_priority, dqn_ver=3, gamma=0.99):
    """
    미니 배치 훈련 : 기억 버퍼(우선도/노멀)에서 랜덤 추출한 경험을 미분/역전파/최적화
    """
    # 1. 데이터 전처리 파트
    # batch 데이터를 언패킹
    states, actions, rewards, next_states, dones = zip(*batch)

    # states를 이미지스택과 시간으로 언패킹
    img_stacks, ticks = zip(*states)
    next_img_stacks, next_ticks = zip(*next_states)

    # # tensor인 states 들은 cat(합침)
    # img_stacks_tensor = torch.cat(img_stacks, dim=0).to(device)  # (32, 4, 64, 64)
    # next_img_stacks_tensor = torch.cat(next_img_stacks, dim=0).to(device)  # (32, 4, 64, 64)

    # 텐서가 아닌 리스트는 tensor로 변환
    # actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
    # 텐서로 변환하는 것보다는 numpy 배열로 만든 후 텐서로 참조만 하는 제로카피가 더 빠름
    # 1. 이미지: CPU에서는 무조건 가벼운 uint8로 묶어서 GPU로 쏜 뒤, GPU 안에서 float으로 변환!
    img_stacks_tensor = torch.as_tensor(np.array(img_stacks, dtype=np.uint8)).to(device).float() / 255.0
    next_img_stacks_tensor = torch.as_tensor(np.array(next_img_stacks, dtype=np.uint8)).to(device).float() / 255.0

    # 2. 시간(ticks): CPU에서부터 float32로 묶어서 Zero-copy 전송
    ticks_tensor = torch.as_tensor(np.array(ticks, dtype=np.float32)).unsqueeze(1).to(device)
    next_ticks_tensor = torch.as_tensor(np.array(next_ticks, dtype=np.float32)).unsqueeze(1).to(device)

    # 3. 행동(actions): Q-Value 인덱싱(gather)을 위해 정수형 int64 (long)로 묶기
    actions_tensor = torch.as_tensor(np.array(actions, dtype=np.int64)).unsqueeze(1).to(device) 
    
    # 4. 보상(rewards)과 종료여부(dones): 벨만 방정식 연산을 위해 float32로 묶기
    rewards_tensor = torch.as_tensor(np.array(rewards, dtype=np.float32)).unsqueeze(1).to(device)
    dones_tensor = torch.as_tensor(np.array(dones, dtype=np.float32)).unsqueeze(1).to(device)

    # 2. 훈련 로직(feat. 벨만 방정식)
    """
    Target = R + r * maxQ(s', a') or R if done
    정답지 = 현재행동의보상 + 할인율 * 최대미래가치(최선의 행동을 했을 때)
    다음 상태에서 사망이면 미래가치는 0
    즉 최대수령가능 보상을 정답지로 두고, 현재 얻은 q값과의 오차를 최대한 줄이는 방향으로 역전파
    """
    # states 텐서 시간 결합
    states_tensor = (img_stacks_tensor, ticks_tensor)
    next_states_tensor = (next_img_stacks_tensor, next_ticks_tensor)
    
    # 현재 상태의 q밸류 쌍 확인
    q_values = model(states_tensor,)  # (32, 3)
    # 그중에 실제로 수행한 action들의 q밸류(gather로 행동별 인덱스의 q값만 추출)
    # q_values: [[1.2, 0.5], [0.1, 0.9], ...] (배치 사이즈 32)
    # actions:  [[0],        [1],        ...] (내가 했던 행동들)
    # acted_q:  [[1.2],      [0.9],      ...] (내가 했던 행동의 q값들)
    acted_q = q_values.gather(dim=1, index=actions_tensor)  # (32, 1)
    # avg_q = acted_q.mean().item()

    # 미래에 획득할 가치(수치확인만이 목적이므로 가중치 수정이 안되도록 기울기 추적을 끊는다)
    with torch.no_grad():
        # 0. 초기 dqn : 현행 모델이 다음상태를 계산하고 최대가치를 직접 계산 -> 가중치 변경 때문에 목표가 끊임없이 움직이는 문제
        if dqn_ver == 0:
            next_q_values = model(next_states_tensor)
            max_next_q_values = next_q_values.max(dim=1, keepdim=True)[0]
        # 1. nature dqn : target이 다음상태의 가치를 계산 -> 고정된 과녁이나, 타겟이 과대평가할 가능성 존재
        elif dqn_ver == 1:
            # 미래 가치들 확인
            next_q_values = target_model(next_states_tensor)  # (32, 3)
            # 최대 미래가치를 뽑아냄(keepdim=True로 차원 유지, 안쓴다면 unsqueeze(1)을 붙여줘야함)
            max_next_q_values = next_q_values.max(dim=1, keepdim=True)[0]   #(32, 1)
        # 2. double dqn : 현행 모델이 최선행동을 고르면 타겟모델이 그 가치를 평가 -> 과대평가 가능성 차단
        else:
            # 현행 모델이 다음스텝의 상태를 계산
            online_next_q = model(next_states_tensor)
            # 현행 모델이 최선행동을 선택
            best_actions = online_next_q.argmax(dim=1, keepdim=True)
            # 타겟 모델이 과거의 가중치를 바탕으로 다음 상태를 계산
            target_next_q = target_model(next_states_tensor)
            # 타겟 모델이 현행모델의 최선행동을 평가함
            max_next_q_values = target_next_q.gather(dim=1, index=best_actions)

        # 벨만방정식의 정답지 공식(사망시 미래가치는 증발하는 것을 (1-dones)로 구현)
        # 결정된 행동에 의한 실제보상(reward) + 할인율 * 예측한 최대 미래 가치 * 생존 여부
        # 신경망에 의한 행동 가치의 예측값의 정답지는 (실제 보상값) + 다음 상태의 예측값(부트스트래핑 - 예측값이 정답지???)
        target_q = rewards_tensor + gamma * max_next_q_values * (1 - dones_tensor)
        # 시간차 오차(TD-Error)
        td_errors = (acted_q - target_q).detach()

    # 3. 역전파
    # 손실함수 : 결과를 본 후 계산된 예측값 - 신경망 예측값 -> 시간차 오차(TD_error)
    loss = F.smooth_l1_loss(acted_q, target_q)
    # 기울기 찌꺼기 제거
    optimizer.zero_grad()
    # 역전파
    loss.backward()
    # 기울기 폭발 방지
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # 가중치 업데이트
    optimizer.step()

    # # 하이브리드 듀얼 버퍼 : TD-error가 큰 기억은 우선도 버퍼에 밀어넣기
    # if buffer_type == 2:
    #     # 오차의 절대값을 계산->토치 기울기 계산에서 분리->평탄화->cpu 램으로->넘파이배열
    #     abs_td_errors = torch.abs(td_errors).squeeze(-1).cpu().numpy()
    #     # 오차 크기 상위 10% 정도를 기준점으로 삼음
    #     threshold = np.percentile(abs_td_errors, 90)
    #     # 배치의 기억 중 기준이상의 TD-error값을 가진 기억을 우선도 버퍼로
    #     to_priority = np.where(abs_td_errors >= threshold)[0]
    #     for i in to_priority:
    #         push_to_priority(batch[i])

    # return td_errors.mean().item(), loss.item()
    return td_errors.cpu().numpy(), loss.item()




