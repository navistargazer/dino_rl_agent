'''
chrome의 dino 게임을 플레이하는 인공지능
강화학습(reinforcement learning)
DQN(Deep Q-Learning)
'''
import numpy as np
import time
import torch
import torch.optim as optim
from dqn_cnn import DQN_CNN
from dino_env import DinoEnvironment
from replay_buffer import ReplayBuffer
from train_buffer import train_buffer
import open_game_window as gw






# 4. 동작의 결과에 따른 보상(reward) 부여(사망/생존) - 역전파로 가중치 수정

# 5. 캐릭터가 사망하면 학습 루프 1회 종료

# 6. 누적 보상의 합을 최대화하는 정책(policy)

MONITOR = {'top': 170, 'left': 140, 'width': 300, 'height': 80}
NUM_EPISODES = 1000 # 총 플레이할 게임 판 수
BATCH_SIZE = 32     # 한 번 학습할 때 꺼내볼 과거 경험의 개수

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN_CNN().to(device)           # 아까 만든 뇌 (PyTorch 신경망 모델)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 2. 현재 상태에 맞는 동작(action) 실행 - 동작:3(0-아무것도 안함, 1-점프, 2-숙이기)
def select_action(state, model, epsilon):
    # epsilon보다 작은 경우는 랜덤 행동, 크면 q-value에 의한 행동
    if np.random.rand() < epsilon:
        return np.random.choice([0, 1])
    else:
        with torch.no_grad():               # 가중치 저장 안함(미분x, 역전파x)
            q_values = model.forward(state) # 현재 상태의 q값
            return torch.argmax(q_values).item()    # 가장 큰 값의 인덱스에 있는 숫자(int)를 반환 -> q값이 [0.1, 0.8]인 경우 1번 인덱스의 item인 1을 반환

def update_epsilon(epsilon):
    if (epsilon > 0.05):
        epsilon *= 0.995
    return epsilon



def train_dino_agent():
    # 1. 초기화 (환경, 모델, 메모리 준비)
    env = DinoEnvironment(MONITOR)         # 키보드 제어, 보상 판단을 통제할 객체
    replaybuffer  = ReplayBuffer(capacity=10000)         # 경험을 저장할 커다란 메모리 공간
    
    epsilon = 1.0                 # 초기 탐험 확률 (처음엔 100% 무작위 행동)
    best_score = 0

    for episode in range(NUM_EPISODES):
        # ----------------------------------------------------
        # [게임 한 판(Episode) 시작]
        # ----------------------------------------------------
        state, total_reward, done = env.restart_game()                  # 브라우저 초기화 및 게임 시작
        frame_count = 0
        while not done: 
            # ----------------------------------------------------
            # [1 Step 진행 (매 프레임마다 일어나는 일)]
            # ----------------------------------------------------
            start = time.time()
            # 2. 행동 결정 (뇌를 거치거나 or 무작위 탐험)
            action = select_action(state, model, epsilon) 
            
            # 3. 환경과 상호작용 (키보드 누르고, 스킵 대기하고, 결과 받기)
            # 앞서 말씀드린 time.sleep(0.06) 같은 타이밍 조절이 이 안에서 일어납니다.
            next_state, reward, done = env.step(action)
            
            interval = time.time() - start
            if interval < 0.066:
                time.sleep(0.066 - interval)
            frame_count += 1

            # # 4. 경험 저장 (방금 겪은 일을 메모리에 기록)
            replaybuffer.push(state, action, reward, next_state, done)
            
            # # 5. 모델 학습 (메모리에 데이터가 충분히 쌓이면 무작위로 꺼내서 복습)
            if len(replaybuffer) > BATCH_SIZE:
                training_iteration = min(frame_count, 500)  # 훈련 상한 설정
                for _ in range(training_iteration):
                    batch = replaybuffer.sample(BATCH_SIZE)
                    train_buffer(model, optimizer, batch, device)
            
            # 6. 베스트 모델 저장
            if frame_count > best_score:
                best_score = frame_count
                torch.save(model.state_dict(), 'best_model.pth')
                print('Best model saved!')

            # 6. 상태 업데이트 (다음 스텝을 위해)
            state = next_state
            total_reward += reward
        # ----------------------------------------------------
        # [게임 한 판 종료]
        # ----------------------------------------------------
        print(f"Episode: {episode} | Score: {frame_count} | Total Reward: {total_reward} | Epsilon: {epsilon:.2f}")
        
        # 판이 끝날 때마다 점차 무작위 탐험(epsilon) 확률을 0.5%씩 줄여나감(최저값은 0.05)
        epsilon = update_epsilon(epsilon) 
        time.sleep(1)

if __name__ == "__main__":
    gw.setup_game_window()
    time.sleep(1)
    train_dino_agent()