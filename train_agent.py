"""
chrome의 dino 게임을 플레이하는 인공지능
강화학습(reinforcement learning)
DQN(Deep Q-Network) 알고리즘
"""

import numpy as np
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import config as cf
from environments import DQN, D3QN, Environment
from trainer import train_buffer, ReplayBuffer
import os
from utils import draw_plots

class TrainAgent:
    def __init__(self):
        # 1. 초기화 (환경, 모델, 메모리 준비)
        self.NUM_EPISODES = cf.NUM_EPISODES
        self.BATCH_SIZE = cf.BATCH_SIZE
        self.DQN_VER = cf.DQN_VER
        self.NUM_BUFFER = cf.NUM_BUFFER
        self.FPS = cf.FPS
        self.TAU = cf.TAU
        self.GAMMA = cf.GAMMA

        # 디바이스 설정
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"사용 디바이스: {self.device}")

        # # 강화학습에서는 pytorch의 과도한 멀티스레드에 의한 cpu오버헤드 방지
        # torch.set_num_threads(1)

        # online model - 버전에 따라 DQN(vanilla, nature, double), D3QN(dueling) 선택
        CNN = DQN if self.DQN_VER < 3 else D3QN
        self.model = CNN(num_actions=3, input_pixel=cf.PIXEL).to(self.device)  # 학습자의 두뇌
        # target model - nature DQN부터 타겟 네트워크 가동
        self.target_model = CNN(num_actions=3, input_pixel=cf.PIXEL).to(self.device)  # 목표 신경망
        self.optimizer = optim.Adam(self.model.parameters(), lr=cf.LEARNING_RATE)
        # 키보드 제어, 보상 판단을 통제할 환경 인스턴스
        env = Environment(self.model)
        # 환경 함수 캐싱
        self._get_q_values = env.get_q_values
        self._restart_game = env.restart_game
        self._step = env.step
        
        self.writer = SummaryWriter("runs/tb_log")  # run/dino_ex_1 폴더에 로그가 쌓임
        
        self.frame_time = 1.0 / self.FPS
        
        # 기억 저장용 버퍼
        self.replaybuffer = ReplayBuffer(
            priority_cap=cf.P_BUFFER_SIZE, normal_cap=cf.N_BUFFER_SIZE, priority_ratio=0.5
        )
        
        # 이어서 학습할 경우를 위한 변수
        self.best_score = 0
        self.epsilon = 1.0

        # 학습 이어하기(모델 저장 파일 로드)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_name = f"best_model_DQN{self.DQN_VER}_{self.NUM_BUFFER}Buffer"
        self.model_path = os.path.join(BASE_DIR, f"models/{model_name}.pth")
        print(f"{model_name} 모델 사용,", end=" ")
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.best_score = checkpoint["best_score"]
            self.epsilon = checkpoint["epsilon"]
            print(
                f"이어서 학습 시작 (기존 최고 생존: {self.best_score} / Epsilon: {self.epsilon:.3f})"
            )
        else:
            os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
            self.best_score = 0
            self.epsilon = 1.0
            print("새로운 학습 시작")
        
        # 그래프 저장 경로
        plot_dir = os.path.join(BASE_DIR, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        self.plot_path = os.path.join(plot_dir, f"{model_name}_plot.png")
         
        # dqn 버전이 올라가면 타겟 모델 등장
        if self.DQN_VER > 0:
            # 타겟 모델에 현행 모델의 상태 주입
            self.target_model.load_state_dict(self.model.state_dict())
            # 타겟 모델은 학습 없이 평가모드로
            self.target_model.eval()

    def validate_model(self, num_test=5):
        '''
        훈련 도중 엡실론을 제거한 진짜 실력을 검증하고 베스트 모델을 저장
        epsilon = 0, 버퍼 저장 및 배치 훈련 안함
        ''' 
        # 모델 평가 모드로 전환
        self.model.eval()
        # 생존 시간 기록
        survival_record = []
        _time = time.time
        _sleep = time.sleep
        _argmax = torch.argmax

        # 미분 없이 에피소드 루프
        with torch.no_grad():
            for i in range(num_test):
                state, done = self._restart_game()
                epi_start_time = _time()
                reward_sum = 0.0
                # 단일 에피소드 시작
                while not done:
                    start = _time()
                    # 1. 도델의 최대 Q 값에 의한 행동 결정
                    q_values = self._get_q_values()
                    act_idx = _argmax(q_values).item()
                    # 2. 다음 상태로 진행
                    next_state, reward, done = self._step(act_idx)
                    # 6. 프레임 간격보다 짧은 시간에 끝났다면 기다림
                    interval = _time() - start
                    if interval < self.frame_time:
                        _sleep(self.frame_time - interval)
                    elif interval > self.frame_time + 0.01:
                        print(f"frame delayed: {interval - self.frame_time:.3f}sec")
                    # 7. 상태 업데이트 (다음 스텝을 위해)
                    state = next_state
                    reward_sum += reward

                # 에피소드 종료 생존시간
                survival_time = _time() - epi_start_time
                survival_record.append(survival_time)
                print(".", end="", flush=True)
        
        # 모델 훈련 모드로 복귀
        self.model.train()
        # 최대 생존 시간
        best_score = max(survival_record)
        # 베스트 모델 저장
        if best_score > self.best_score:
            self.best_score = best_score
            checkpoint = {
                        "model_state_dict": self.model.state_dict(),
                        "best_score": self.best_score,
                        "epsilon": self.epsilon,
            }
            torch.save(checkpoint, self.model_path)
            print('베스트 모델이 갱신되었습니다.')
        else:
            print("모델 갱신 실패")

    # 에이전트 훈련 함수
    def train_agent(self):
        """
        훈련 루프

        """
        # 시각화 준비
        history_q_values = []
        history_survived = []

        # while루프 내 라이브러러 캐싱
        _time = time.time
        _max = torch.max
        _argmax = torch.argmax
        _rand = np.random.rand
        _randint = np.random.randint
        _sleep = time.sleep

        _push = self.replaybuffer.push
        _sample = self.replaybuffer.sample
        _add_scalar = self.writer.add_scalar

        half_episode = self.NUM_EPISODES // 2

        # 게임 에피소드 반복 시작
        for episode in range(self.NUM_EPISODES):
            state, done = self._restart_game()  # 브라우저 초기화 및 게임 시작
            # time.sleep(1)
            # q-value 시각화 준비
            epi_start_time = _time()
            epi_frame_cnt = 0
            reward_sum, init_max_q = 0.0, 0.0
            # d3qn 시각화 준비
            if self.DQN_VER == 3:
                sum_value, sum_advantage = 0.0, 0.0

            # 단일 에피소드 시작
            while not done:
                start = _time()
                epi_frame_cnt += 1
                # frame_since_update += 1
                if epi_frame_cnt % 10 == 0:
                    print("#", end="", flush=True)
                # 1. 행동 결정 (뇌를 거치거나 or 무작위 탐험)
                # dueling dqn의 경우 V와 A도 받아서 시각화
                # q값 연선에서는 기울기 계산은 필요 없음(나중에 배치 훈련에서만 역전파 업데이트)
                with torch.no_grad():
                    if self.DQN_VER < 3:
                        q_values = self._get_q_values()
                    else:
                        q_values, val, adv = self._get_q_values(return_dueling=True)
                        avg_value = val.mean().item()
                        sum_value += avg_value
                        avg_advantage = (adv.max(dim=1)[0] - adv.min(dim=1)[0]).mean().item()
                        sum_advantage += avg_advantage
                    max_q = _max(q_values).item()
                    # 에피소드 시작 시 최대 Q값을 시각화
                    if epi_frame_cnt == 1:
                        init_max_q = max_q

                    # epsilon-greedy 행동 결정
                    if _rand() < self.epsilon:
                        act_idx = _randint(3)                # 랜덤(0:대기, 1:점프, 2:숙이기)
                    else:
                        act_idx = _argmax(q_values).item()   # 최대 가지를 가진 행동의 인덱스를 선택

                # 2. 1스텝 진행(다음 상태 확인)
                next_state, reward, done = self._step(act_idx)

                # 3. 경험 저장 (방금 겪은 일을 메모리에 저장, 우선도/노멀 듀얼버퍼)
                _push((state, act_idx, reward, next_state, done), self.NUM_BUFFER)

                # ===== 미니 배치 훈련(미분 및 역전파) ======
                # 4. 모델 학습 (메모리에 데이터가 충분히 쌓이면 무작위로 꺼내서 복습)
                if len(self.replaybuffer) > self.BATCH_SIZE:
                    epi_progress = episode / self.NUM_EPISODES
                    batch = _sample(self.BATCH_SIZE, self.NUM_BUFFER, epi_progress)
                    train_buffer(
                        self.model, self.target_model, self.optimizer, batch, self.device, self.DQN_VER, self.GAMMA
                    )
                    # 매 스텝마다 타겟 네트워크를 소프트 업데이트
                    for target_param, online_param in zip(
                        self.target_model.parameters(), self.model.parameters()
                    ):
                        target_param.data.copy_(
                            self.TAU * online_param.data + (1.0 - self.TAU) * target_param.data
                        )

                # 6. 프레임 간격보다 짧은 시간에 끝났다면 기다림
                interval = _time() - start
                if interval < self.frame_time:
                    _sleep(self.frame_time - interval)
                elif interval > self.frame_time + 0.01:
                    print(f"frame delayed: {interval - self.frame_time:.3f}sec")

                # 7. 상태 업데이트 (다음 스텝을 위해)
                state = next_state
                reward_sum += reward

            # 에피소드 종료 생존시간
            survival_time = _time() - epi_start_time
            # 결과 출력
            print(
                f"\nEpisode {episode:4d} | Survived {survival_time:.3f} | Max_Q {init_max_q:.2f} | Total Reward {reward_sum:.2f} | Epsilon {self.epsilon:.2f}"
            )

            # 베스트 모델 저장 & 학습률 소프트 업데이트
            if (episode + 1) % 100 == 0:
                print("모델 검증 중", end="")
                self.validate_model(num_test=5)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = max(1e-5, param_group["lr"] * 0.95)
                print(f"학습률 감소 : {self.optimizer.param_groups[0]['lr']:.7f}")
            elif (episode > half_episode) and (survival_time > self.best_score * 1.2):
                print("기존 기록 20% 이상 갱신, 모델 검증 중", end="")
                self.validate_model(num_test=5)

            # 판이 끝날 때마다 점차 무작위 탐험(epsilon) 확률을 0.5%씩 줄여나감(최저값은 0.05)
            self.epsilon = max(cf.EPSILON_MIN, self.epsilon * cf.EPSILON_DECAY)

            # 텐서보드 로그
            _add_scalar("Performance/1_Survival_Time", survival_time, episode)
            _add_scalar("Performance/2_Total_Reward", reward_sum, episode)
            _add_scalar("Brain/Max Q-Value", init_max_q, episode)
            # writer.flush()
            history_q_values.append(init_max_q)
            history_survived.append(survival_time)
            # if episode % 10 == 0:
            draw_plots(history_survived, history_q_values, self.plot_path)

        # 훈련 종료
        self.writer.close()


if __name__ == "__main__":
    trainer = TrainAgent()
    trainer.train_agent()