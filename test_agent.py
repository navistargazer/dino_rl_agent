'''
chrome의 dino 게임을 플레이하는 인공지능
강화학습(reinforcement learning)
DQN(Deep Q-Learning)
'''
import time
import torch
import config as cf
from dqn_cnn import DQN_CNN
from dino_env import DinoEnvironment
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = DinoEnvironment()         # 키보드 제어, 보상 판단을 통제할 객체
model = DQN_CNN(num_actions=3).to(device)        # 학습자의 두뇌

def test_agent():
    # 1. 초기화 (환경, 모델, 메모리 준비)
    model_path = 'models/best_model.pth'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    epsilon = 0.0
    for episode in range(cf.NUM_EPISODES):
        state, total_reward, done = env.restart_game()               # 브라우저 초기화 및 게임 시작
        epi_start_time = time.time()

        while not done: 
            start = time.time()
            # 1. 행동 결정 (뇌를 거치거나 or 무작위 탐험)
            action = env.select_action(model, epsilon) 
                        
            # 2. 1스텝 진행(다음 상태 확인)
            next_state, reward, done = env.step(action)

            # 6. 프레임 간격보다 짧은 시간에 끝났다면 기다림
            interval = time.time() - start
            if interval < cf.FRAME_INTERVAL:
                time.sleep(cf.FRAME_INTERVAL - interval)
            elif interval > cf.FRAME_INTERVAL + 0.01:
                print(f'frame delayed: {interval - cf.FRAME_INTERVAL:.2f}sec')

            # 7. 상태 업데이트 (다음 스텝을 위해)
            state = next_state
            total_reward += reward

        survival_time = time.time() - epi_start_time

        print(f"Episode: {episode} | Survived: {survival_time:.2f} | Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    test_agent()