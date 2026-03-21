import torch
import torch.nn.functional as F

def train_buffer(model, optimizer, batch, device):
    # 1. 데이터 전처리 파트
    # batch 데이터를 언패킹
    states, actions, rewards, next_states, dones = zip(*batch)

    # tensor인 states 들은 cat(합침)
    states_tensor = torch.cat(states, dim=0).to(device)             # (32, 4, 84, 84)
    next_states_tensor = torch.cat(next_states, dim=0).to(device)    # (32, 4, 84, 84)

    # 나머지는 tensor로 변환
    actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)   # 나중에 q밸류 인덱싱을 위해 int64로
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device) 
    dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)     # 벨만방정식 연산을 위해 float32로

    # 2. 훈련 로직(feat. 벨만 방정식)
    '''
    Target = R + r * maxQ(s', a') or R if done
    정답지 = 현재행동의보상 + 할인율 * 최대미래가치(최선의 행동을 했을 때)
    다음 상태에서 사망이면 미래가치는 0
    즉 최대수령가능 보상을 정답지로 두고, 현재 얻은 q값과의 오차를 최대한 줄이는 방향으로 역전파
    '''
    # 현재 상태의 q밸류 쌍 확인
    q_values = model(states_tensor)                          # (32, 2)
    # 그중에 실제로 수행한 action들의 q밸류(gaher로 행동별 인덱스의 q값만 추출)
    acted_q = q_values.gather(dim=1, index=actions_tensor)  # (32, 1)


    # 미래에 획득할 가치(수치확인만이 목적이므로 가중치 수정이 안되도록 기울기 추적을 끊는다)
    with torch.no_grad():
        # 미래 가치들 확인
        next_q_values = model(next_states_tensor)           # (32, 2)
        # 최대 미래가치를 뽑아냄(keepdim=True로 차원 유지, 안쓴다면 unsqueeze(1)을 붙여줘야함)
        max_next_q_values = next_q_values.max(dim=1, keepdim=True)[0]
        # 벨만방정식의 정답지 공식(사망시 미래가치는 증발하는 것을 (1-dones)로 구현)
        target_q = rewards_tensor + 0.99 * max_next_q_values * (1 - dones_tensor)
    
    # 3. 역전파
    # 손실함수
    loss = F.mse_loss(acted_q, target_q)
    # 기울기 찌꺼기 제거
    optimizer.zero_grad()
    # 역전파
    loss.backward()
    # 가중치 업데이트
    optimizer.step()    





