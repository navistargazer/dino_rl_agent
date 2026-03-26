'''
chrome의 dino 게임을 플레이하는 인공지능
강화학습(reinforcement learning)
DQN(Deep Q-Learning)
'''
import numpy as np
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import config as cf
from dqn_cnn import DQN_CNN
from dino_env import DinoEnvironment
from replay_buffer import ReplayBuffer
from train_buffer import train_buffer
import os
import analysis

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available(): # ⭐️ M1/M2 맥을 위한 가속 장치!
    device = torch.device("mps")
    print("Mac GPU (MPS) 가동!")
else:
    device = torch.device("cpu")
env = DinoEnvironment()         # 키보드 제어, 보상 판단을 통제할 객체
model = DQN_CNN(num_actions=3).to(device)        # 학습자의 두뇌
target_model = DQN_CNN(num_actions=3).to(device) # 목표 신경망
optimizer = optim.Adam(model.parameters(), lr=cf.LEARNING_RATE)
writer = SummaryWriter('runs/dino_ex_1')    # run/dino_ex_1 폴더에 로그가 쌓임

def train_agent():
    # 1. 초기화 (환경, 모델, 메모리 준비)
    replaybuffer  = ReplayBuffer(priority_cap=cf.P_BUFFER_SIZE, normal_cap=cf.N_BUFFER_SIZE, priority_ratio=0.5)         # 경험을 저장할 커다란 메모리 공간
    best_score = 0
    epsilon = 1.0
    total_steps = 0

    
    # 학습 이어하기(모델 저장 파일 로드)
    model_path = 'models/best_model.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_score = checkpoint['best_score']
        epsilon = checkpoint['epsilon']
        print(f'이어서 학습 시작 (기존 최고 생존: {best_score} / Epsilon: {epsilon:.3f})')
    else:
        best_score = 0
        epsilon = 1.0
        print('새로운 학습 시작')

    # 타겟 모델에 모델의 상태 저장
    target_model.load_state_dict(model.state_dict())
    # 타겟 모델은 학습 없이 평가모드로
    target_model.eval()

    history_q_values = []
    history_reward = []

    for episode in range(cf.NUM_EPISODES):
        state, done = env.restart_game()               # 브라우저 초기화 및 게임 시작
        # time.sleep(1)
        # q-value 시각화 준비
        epi_start_time = time.time()
        epi_frame_cnt = 0
        reward_sum = 0.0
        q_value_sum = 0.0
        while not done: 
            start = time.time()
            # 1. 행동 결정 (뇌를 거치거나 or 무작위 탐험)
            q_values = env.get_q_values(model)
            max_q = torch.max(q_values).item()
            q_value_sum += max_q

            if np.random.rand() < epsilon:
                action = np.random.randint(3)
            else:
                action = torch.argmax(q_values).item()
                        
            # 2. 1스텝 진행(다음 상태 확인)
            epi_frame_cnt += 1
            next_state, reward, done = env.step(action)
            
            # 3. 경험 저장 (방금 겪은 일을 메모리에 기록)
            replaybuffer.push((state, action, reward, next_state, done))

            # 4. 모델 학습 (메모리에 데이터가 충분히 쌓이면 무작위로 꺼내서 복습)
            if len(replaybuffer) > cf.BATCH_SIZE:
                batch = replaybuffer.sample(cf.BATCH_SIZE)
                train_buffer(model, target_model, optimizer, batch, device)

            # 5. 1000 프레임마다 타겟모델 업데이트
            total_steps += 1
            if (total_steps % cf.UPDATE_FREQ) == 0:
                target_model.load_state_dict(model.state_dict())
                # print('타겟 모델 업데이트')

            # 6. 프레임 간격보다 짧은 시간에 끝났다면 기다림
            interval = time.time() - start
            if interval < cf.FRAME_INTERVAL:
                time.sleep(cf.FRAME_INTERVAL - interval)
            elif interval > cf.FRAME_INTERVAL + 0.01:
                print(f'frame delayed: {interval - cf.FRAME_INTERVAL:.2f}sec')

            # 7. 상태 업데이트 (다음 스텝을 위해)
            state = next_state
            reward_sum += reward
        # 에피소드 종료
        # 생존시간
        survival_time = time.time() - epi_start_time
        
        # 베스트 모델 저장
        if survival_time > best_score:
            best_score = survival_time
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'best_score': best_score,
                'epsilon': epsilon,
            }
            torch.save(checkpoint, 'models/best_model.pth')
            print('Best model saved!')

        
        # 판이 끝날 때마다 점차 무작위 탐험(epsilon) 확률을 0.5%씩 줄여나감(최저값은 0.05)
        epsilon = max(cf.EPSILON_MIN, epsilon * cf.EPSILON_DECAY)

        # 텐서보드 로그
        avg_q_values = q_value_sum / epi_frame_cnt
        writer.add_scalar('Performance/1_Survival_Time', survival_time, episode)
        writer.add_scalar('Performance/2_Total_Reward', reward_sum, episode)
        writer.add_scalar('Brain/Average Q-Value', avg_q_values, episode)
        # writer.flush()
        # 결과 출력
        print(f"Episode: {episode} | Survived: {survival_time:.2f} | AVG_Q: {avg_q_values:.2f} | Total Reward: {reward_sum:.2f} | Epsilon: {epsilon:.2f}")
        history_q_values.append(avg_q_values)
        history_reward.append(reward_sum)
        # if episode % 10 == 0:
        analysis.visualize_q_values(history_reward, history_q_values)
    writer.close()
        
if __name__ == "__main__":
    train_agent()