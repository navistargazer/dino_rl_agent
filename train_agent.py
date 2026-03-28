"""
chrome의 dino 게임을 플레이하는 인공지능
강화학습(reinforcement learning)
DQN(Deep Q-Learning)
"""

import numpy as np
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import config as cf
from environments import DQN_CNN, DinoEnvironment
from trainer import train_buffer, ReplayBuffer
import os
import utils.visualize as visualize


def train_agent():
    # 1. 초기화 (환경, 모델, 메모리 준비)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # ⭐️ M1/M2 맥을 위한 가속 장치!
        device = torch.device("mps")
        print("Mac GPU (MPS) 가동!")
    else:
        device = torch.device("cpu")
        print("CPU 가동!")

    # # 강화학습에서는 pytorch의 과도한 멀티스레드에 의한 cpu오버헤드 방지
    # torch.set_num_threads(1)

    env = DinoEnvironment()  # 키보드 제어, 보상 판단을 통제할 객체
    # online model
    model = DQN_CNN(num_actions=3, input_pixel=cf.PIXEL).to(device)  # 학습자의 두뇌
    # target model
    target_model = DQN_CNN(num_actions=3, input_pixel=cf.PIXEL).to(
        device
    )  # 목표 신경망

    optimizer = optim.Adam(model.parameters(), lr=cf.LEARNING_RATE)

    writer = SummaryWriter("runs/dino_ex_1")  # run/dino_ex_1 폴더에 로그가 쌓임
    frame_time = 1.0 / cf.FPS
    replaybuffer = ReplayBuffer(
        priority_cap=cf.P_BUFFER_SIZE, normal_cap=cf.N_BUFFER_SIZE, priority_ratio=0.5
    )  # 경험을 저장할 커다란 메모리 공간
    best_score = 0
    epsilon = 1.0
    # 타겟 모델 업데이트 변수(1000프레임 이상, 3에피소드 이상)
    frame_since_update = 0
    episode_since_update = 0

    # 학습 이어하기(모델 저장 파일 로드)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(
        BASE_DIR, f"best_model_DQN{cf.DQN_VER}_{cf.NUM_BUFFER}Buffer.pth"
    )
    print(f"DQN{cf.DQN_VER}_{cf.NUM_BUFFER}Buffer 모델 사용", end=" ")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_score = checkpoint["best_score"]
        epsilon = checkpoint["epsilon"]
        print(
            f"이어서 학습 시작 (기존 최고 생존: {best_score} / Epsilon: {epsilon:.3f})"
        )
    else:
        best_score = 0
        epsilon = 1.0
        print("새로운 학습 시작")
    # dqn 버전이 올라가면 타겟 모델 등장
    if cf.DQN_VER > 0:
        # 타겟 모델에 모델의 상태 저장
        target_model.load_state_dict(model.state_dict())
        # 타겟 모델은 학습 없이 평가모드로
        target_model.eval()

    # 시각화 준비
    history_q_values = []
    history_survived = []

    # while루프 내 라이브러러 캐싱
    _NUM_EPISODES = cf.NUM_EPISODES
    _BATCH_SIZE = cf.BATCH_SIZE
    _UPDATE_FREQ = cf.UPDATE_FREQ
    _DQN_VER = cf.DQN_VER
    _NUM_BUFFER = cf.NUM_BUFFER

    _get_q_values = env.get_q_values
    _restart_game = env.restart_game
    _step = env.step

    _max = torch.max
    _argmax = torch.argmax
    _rand = np.random.rand
    _randint = np.random.randint
    _time = time.time
    _sleep = time.sleep

    _push = replaybuffer.push
    _sample = replaybuffer.sample
    _add_scalar = writer.add_scalar
    _draw_plots = visualize.draw_plots

    # 게임 에피소드 반복 시작
    for episode in range(_NUM_EPISODES):
        state, done = _restart_game()  # 브라우저 초기화 및 게임 시작
        # time.sleep(1)
        # q-value 시각화 준비
        epi_start_time = _time()
        epi_frame_cnt = 1
        reward_sum = 0.0
        first_q = 0.0

        # 단일 에피소드 시작
        while not done:
            start = _time()
            frame_since_update += 1
            if epi_frame_cnt % 10 == 0:
                print("#", end="", flush=True)
            # 1. 행동 결정 (뇌를 거치거나 or 무작위 탐험)
            q_values = _get_q_values(model)
            max_q = _max(q_values).item()
            # 에피소드 시작 시 최대 Q값을 시각화
            if epi_frame_cnt == 1:
                first_q = max_q
            if _rand() < epsilon:
                action = _randint(3)
            else:
                action = _argmax(q_values).item()

            # 2. 1스텝 진행(다음 상태 확인)
            epi_frame_cnt += 1
            next_state, reward, done = _step(action)

            # 3. 경험 저장 (방금 겪은 일을 메모리에 기록)
            _push((state, action, reward, next_state, done), _NUM_BUFFER)

            # 4. 모델 학습 (메모리에 데이터가 충분히 쌓이면 무작위로 꺼내서 복습)
            if len(replaybuffer) > _BATCH_SIZE:
                batch = _sample(_BATCH_SIZE, _NUM_BUFFER)
                train_buffer(model, target_model, optimizer, batch, device, _NUM_BUFFER)

            # 6. 프레임 간격보다 짧은 시간에 끝났다면 기다림
            interval = _time() - start
            if interval < frame_time:
                _sleep(frame_time - interval)
            elif interval > frame_time + 0.01:
                print(f"frame delayed: {interval - frame_time:.3f}sec")

            # 7. 상태 업데이트 (다음 스텝을 위해)
            state = next_state
            reward_sum += reward

        # 에피소드 종료
        episode_since_update += 1
        # 타겟 모델 업데이트 후 1000프레임 이상, 3에피소드 이상인 경우 타겟 모델 업데이트
        if frame_since_update > _UPDATE_FREQ and episode_since_update >= 3:
            target_model.load_state_dict(model.state_dict())
            print(
                f"타겟 모델 업데이트 after {episode_since_update}episodes, {frame_since_update}frames"
            )
            frame_since_update = 0
            episode_since_update = 0
        # 생존시간
        survival_time = _time() - epi_start_time

        # 베스트 모델 저장
        if (episode > _NUM_EPISODES - 100) and survival_time > best_score:
            best_score = survival_time
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "best_score": best_score,
                "epsilon": epsilon,
            }
            os.makedirs("models", exist_ok=True)
            torch.save(
                checkpoint,
                os.path.join(
                    BASE_DIR,
                    f"models/best_model_DQN{cf.DQN_VER}_{cf.NUM_BUFFER}Buffer.pth",
                ),
            )
            print("\nBest model saved!")
        else:
            print()

        if episode > 0 and episode % 100 == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.95
            print(f"학습률 감소 : {optimizer.param_groups[0]['lr']:.7f}")

        # 판이 끝날 때마다 점차 무작위 탐험(epsilon) 확률을 0.5%씩 줄여나감(최저값은 0.05)
        epsilon = max(cf.EPSILON_MIN, epsilon * cf.EPSILON_DECAY)

        # 텐서보드 로그
        _add_scalar("Performance/1_Survival_Time", survival_time, episode)
        _add_scalar("Performance/2_Total_Reward", reward_sum, episode)
        _add_scalar("Brain/Max Q-Value", first_q, episode)
        # writer.flush()
        # 결과 출력
        print(
            f"Episode: {episode} | Survived: {survival_time:.2f} | Max_Q: {first_q:.2f} | Total Reward: {reward_sum:.2f} | Epsilon: {epsilon:.2f}"
        )
        history_q_values.append(first_q)
        history_survived.append(survival_time)
        # if episode % 10 == 0:
        _draw_plots(history_survived, history_q_values)
    writer.close()


if __name__ == "__main__":
    train_agent()
