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
from environments import DQN, D3QN, Environment
from trainer import train_buffer, ReplayBuffer
import os
from utils import draw_plots

from dataclasses import dataclass
from enum import IntEnum

# 하이퍼 파라미터 열겨형 정의
class DQNType(IntEnum):
    VANILLA = 0
    NATURE = 1
    DOUBLE = 2
    DUELING = 3

class ImgProcess(IntEnum):
    NONE = 0
    CANNY = 1
    DIFF = 2

class BufferType(IntEnum):
    SINGLE = 0
    DUAL = 1
    HYBRID = 2

class TargetUpdate(IntEnum):
    HARD = 0
    SOFT = 1

@dataclass
class HyperParameters:
    dqn_type: DQNType = DQNType.DUELING
    img_process: ImgProcess = ImgProcess.DIFF
    buffer_type: BufferType = BufferType.HYBRID
    target_update: TargetUpdate = TargetUpdate.SOFT
    pixel: int = 64
    num_episodes: int = 1000
    batch_size: int = 32
    p_ratio: float = 0.5
    p_buffer_size: int = 10000
    n_buffer_size: int = 40000
    fps: int = 15
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    update_freq: int = 1000
    tau: float = 0.005
    learning_rate: float = 0.0001
    gamma: float = 0.99
    without_log: bool = False
    # 들어온 값을 enum 오브젝트로 캐스팅
    def __post_init__(self):
        self.dqn_type = DQNType(self.dqn_type)
        self.img_process = ImgProcess(self.img_process)
        self.buffer_type = BufferType(self.buffer_type)
        self.target_update = TargetUpdate(self.target_update)


class TrainAgent:
    def __init__(self, hp: HyperParameters = HyperParameters(), logging=True):
        # DQN 버전 (0:vanilla, 1:nature, 2:double, 3:dueling )        
        self.DQN_Type = hp.dqn_type
        # 화면 이미지 전처리 방식 (0:흑백만, 1:윤곽선검출, 2:프레임차이(difference))
        self.IMG_PROCESS_TYPE = hp.img_process
        # 기억용 버퍼 타입 (0:단일버퍼, 1:우선도+노멀 듀얼버퍼, 2:하이브리드 듀얼 버퍼)
        self.BUFFER_TYPE = hp.buffer_type
        # 타겟 네트워크 업데이트 타입 (0:1000프레임마다 하드 업데이트, 1:소프트 업데이트)
        self.TARGET_UPDATE = hp.target_update
        # CNN 인풋용 리사이즈 픽셀크기
        self.PIXEL = hp.pixel
        # 훈련 반복 수
        self.NUM_EPISODES = hp.num_episodes
        # 미니 배치 훈련에 사용할 과거 경험 개수
        self.BATCH_SIZE = hp.batch_size
        # 우선도 버퍼에서 꺼내는 비율 (초기값, 0.2까지 줄어들도록 설계)
        self.P_RATIO = hp.p_ratio
        # 우선도 버퍼 크기
        self.P_BUFFER_SIZE = hp.p_buffer_size
        # 우선도 버퍼 크기
        self.N_BUFFER_SIZE = hp.n_buffer_size
        # 우선도 버퍼 크기
        self.FPS = hp.fps
        # 엡실론 최소값
        self.EPSILON_MIN = hp.epsilon_min
        # 엡실론 감소율
        self.EPSILON_DECAY = hp.epsilon_decay
        # 타겟 네트워크 업데이트 주기
        self.UPDATE_FREQ = hp.update_freq
        # 타겟 네트워크 소프트 업데이트 비율
        self.TAU = hp.tau
        # 학습률
        self.LEARNING_RATE = hp.learning_rate
        # 미래가치 할인율
        self.GAMMA = hp.gamma
        self.LOGGING = not hp.without_log
        print(f'[INFO] DQN:{self.DQN_Type.name}, buffer:{self.BUFFER_TYPE.name}, ImgProcess:{self.IMG_PROCESS_TYPE.name}, TargetUpdate:{self.TARGET_UPDATE.name}')

        # 디바이스 설정
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.xprint(f"사용 디바이스: {self.device}")

        # 강화학습에서는 pytorch의 과도한 멀티스레드에 의한 cpu오버헤드 방지
        torch.set_num_threads(1)

        # online model - 버전에 따라 DQN(vanilla, nature, double), D3QN(dueling) 선택
        CNN = DQN if self.DQN_Type < DQNType.DUELING else D3QN
        self.model = CNN(num_actions=3, input_pixel=self.PIXEL).to(self.device)  # 학습자의 두뇌
        # target model - nature DQN부터 타겟 네트워크 가동
        self.target_model = CNN(num_actions=3, input_pixel=self.PIXEL).to(self.device)  # 목표 신경망
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        # 키보드 제어, 보상 판단을 통제할 환경 인스턴스
        self.env = Environment(self.IMG_PROCESS_TYPE, self.PIXEL, logging=self.LOGGING)
        # 환경 함수 캐싱
        self._restart_game = self.env.restart_game
        self._step = self.env.step
        
        
        self.frame_time = 1.0 / self.FPS
        
        # 기억 저장용 버퍼
        self.replaybuffer = ReplayBuffer(
            priority_cap=self.P_BUFFER_SIZE, normal_cap=self.N_BUFFER_SIZE, priority_ratio=self.P_RATIO
        )
        
        # 이어서 학습할 경우를 위한 변수
        self.best_score = 0
        self.epsilon = 1.0

        # 학습 이어하기(모델 저장 파일 로드)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.model_name = f"best_model_{self.DQN_Type.name}_{self.BUFFER_TYPE.name}_{self.IMG_PROCESS_TYPE.name}_{self.TARGET_UPDATE.name}"
        self.model_path = os.path.join(BASE_DIR, f"models/{self.model_name}.pth")
        self.xprint(f"{self.model_name} 모델 사용,", end=" ")
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.best_score = checkpoint["best_score"]
            self.epsilon = checkpoint["epsilon"]
            self.xprint(
                f"이어서 학습 시작 (기존 최고 생존: {self.best_score} / Epsilon: {self.epsilon:.3f})"
            )
        else:
            os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
            self.best_score = 0
            self.epsilon = 1.0
            self.xprint("새로운 학습 시작")
        
        # 그래프 저장 경로
        plot_dir = os.path.join(BASE_DIR, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        self.plot_path = os.path.join(plot_dir, f"{self.model_name}_plot.png")
        # tensorboard 로그 경로
        log_dir = os.path.join(BASE_DIR, "runs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{self.model_name}_log")
        self.writer = SummaryWriter(log_path)  # run/self.model_name 폴더에 로그가 쌓임
        # dqn 버전이 올라가면 타겟 모델 등장
        if self.DQN_Type > 0:
            # 타겟 모델에 현행 모델의 상태 주입
            self.target_model.load_state_dict(self.model.state_dict())
            # 타겟 모델은 학습 없이 평가모드로
            self.target_model.eval()

        self.history_q_values = []
        self.history_survived = []

    def validate_model(self, num_test=5):
        '''
        훈련 도중 엡실론을 제거한 진짜 실력을 검증하고 베스트 모델을 저장
        epsilon = 0, 버퍼 저장 및 배치 훈련 안함
        ''' 
        # 모델 평가 모드로 전환
        self.model.eval()
        # 생존 시간 기록
        survival_record = []
        _argmax = torch.argmax
        _max = torch.max

        # 시각화 준비

        # 미분 없이 에피소드 루프
        with torch.no_grad():
            for i in range(num_test):
                state, done = self._restart_game()
                epi_start_time = time.time()
                reward_sum = 0.0
                epi_frame_cnt = 0
                # 단일 에피소드 시작
                while not done:
                    start = time.time()
                    epi_frame_cnt += 1
                    # 1. 도델의 최대 Q 값에 의한 행동 결정
                    q_values = self.model(state.to(self.device))
                    act_idx = _argmax(q_values).item()

                    if epi_frame_cnt == 1:
                        max_q = _max(q_values).item()
                    if epi_frame_cnt % 10 == 0:
                        self.xprint("#", end="", flush=True)
                    
                    # 2. 다음 상태로 진행
                    next_state, reward, done = self._step(act_idx)
                    # 6. 프레임 간격보다 짧은 시간에 끝났다면 기다림
                    interval = time.time() - start
                    if interval < self.frame_time:
                        time.sleep(self.frame_time - interval)
                    elif interval > self.frame_time + 0.01:
                        self.xprint(f"frame delayed: {interval - self.frame_time:.3f}sec")
                    reward_sum += reward

                # 에피소드 종료 생존시간
                survival_time = time.time() - epi_start_time
                # 기록
                self.history_q_values.append(max_q)
                self.history_survived.append(survival_time)
                # 결과 출력
                self.xprint(
                    f"\nTest{i} | Survived {survival_time:.3f} | Max_Q {max_q:.3f} | Total Reward {reward_sum:.2f}"
                )
                survival_record.append(survival_time)
        
        # 모델 훈련 모드로 복귀
        self.model.train()
        # 그래프 저장
        draw_plots(self.history_survived, self.history_q_values, self.plot_path)

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
            self.xprint('베스트 모델이 갱신되었습니다.')
        else:
            self.xprint("모델 갱신 실패")

    # 에이전트 훈련 함수
    def train_agent(self):
        """
        에피소드 수 만큼 반복 훈련
        1. 현재 프레임(상태)에서 최대 Q값에 의한 행동 선택
        2. 다음 state 정보를 받음
        3. 행동/상태 결과를 메모리 버퍼에 저장
        4. 버퍼에서 랜덤으로 기억을 꺼내서 역전파 학습
        5. 다음 프레임으로 넘어감
        6. 에피소드 종료 시 학습률, 타겟 네트워크, 엡실론 갱신(점진적으로 감소)
        7. 텐서보드로 로깅
        """

        # while루프 내 라이브러러 캐싱
        _max = torch.max
        _argmax = torch.argmax
        _rand = np.random.rand
        _randint = np.random.randint

        _push = self.replaybuffer.push
        _sample = self.replaybuffer.sample
        _push_to_priority = self.replaybuffer.push_to_priority

        _add_scalar = self.writer.add_scalar

        half_episode = self.NUM_EPISODES // 2
        frame_since_update = 0
        episode_since_update = 0
        # 게임 에피소드 반복 시작
        for episode in range(self.NUM_EPISODES):
            # 최초 상태를 가져옴, 브라우저 초기화 및 게임 시작
            state, done = self._restart_game()
            # time.sleep(1)
            # q-value 시각화 준비
            epi_start_time = time.time()
            epi_frame_cnt = 0
            reward_sum, max_q = 0.0, 0.0
            # d3qn 시각화 준비
            if self.DQN_Type == DQNType.DUELING:
                sum_value, sum_advantage = 0.0, 0.0

            # 단일 에피소드 시작
            while not done:
                start = time.time()
                epi_frame_cnt += 1
                frame_since_update += 1
                # 1. 행동 결정 (뇌를 거치거나 or 무작위 탐험)
                # dueling dqn의 경우 V와 A도 받아서 시각화
                # q값 연선에서는 기울기 계산은 필요 없음(나중에 배치 훈련에서만 역전파 업데이트)
                with torch.no_grad():
                    # 현재 state를 device에 올리고 신경망 모델에서 q값을 받아옴
                    # dueling에서는 V값과 A값을 받아와서 시각화 등이 가능함.
                    if self.DQN_Type == DQNType.DUELING:
                        q_values, val, adv = self.model(state.to(self.device), return_dueling=True)
                        avg_value = val.mean().item()
                        sum_value += avg_value
                        avg_advantage = (adv.max(dim=1)[0] - adv.min(dim=1)[0]).mean().item()
                        sum_advantage += avg_advantage
                    else:
                        q_values = self.model(state.to(self.device))

                    # epsilon-greedy 행동 결정
                    if _rand() < self.epsilon:
                        act_idx = _randint(3)                # 랜덤(0:대기, 1:점프, 2:숙이기)
                    else:
                        act_idx = _argmax(q_values).item()   # 최대 가지를 가진 행동의 인덱스를 선택
                    
                    # 에피소드 시작 시 최대 Q값을 시각화
                    if epi_frame_cnt == 1:
                        max_q = _max(q_values).item()

                # 2. 1스텝 진행(다음 상태 확인)
                next_state, reward, done = self._step(act_idx)

                # 3. 경험 저장 (방금 겪은 일을 메모리에 저장, 우선도/노멀 듀얼버퍼)
                _push((state, act_idx, reward, next_state, done), self.BUFFER_TYPE)

                # ===== 미니 배치 훈련(미분 및 역전파) ======
                # 4. 모델 학습 (메모리에 데이터가 충분히 쌓이면 무작위로 꺼내서 복습)
                if len(self.replaybuffer) > self.BATCH_SIZE:
                    epi_progress = episode / self.NUM_EPISODES
                    batch = _sample(self.BATCH_SIZE, self.BUFFER_TYPE, epi_progress)
                    train_buffer(
                        self.model, self.target_model, self.optimizer, batch, self.device, _push_to_priority, self.DQN_Type, self.GAMMA
                    )
                    # 타겟 네트워크 소프트 업데이트
                    if self.TARGET_UPDATE == TargetUpdate.SOFT:
                        # 매 스텝마다 타겟 네트워크를 소프트 업데이트
                        for target_param, online_param in zip(
                            self.target_model.parameters(), self.model.parameters()
                        ):
                            target_param.data.copy_(
                                self.TAU * online_param.data + (1.0 - self.TAU) * target_param.data
                            )

                # 6. 프레임 간격보다 짧은 시간에 끝났다면 기다림
                interval = time.time() - start
                if interval < self.frame_time:
                    time.sleep(self.frame_time - interval)
                elif interval > self.frame_time + 0.01:
                    self.xprint(f"frame delayed: {interval - self.frame_time:.3f}sec")

                # 7. 다음 상태를 저장(기억 버퍼에 현재 상태로 전달되는 임시 변수 역할)
                state = next_state
                reward_sum += reward
            # 렉걸릴 때는 그냥 넘김
            if epi_frame_cnt < 1:
                print("#", end="", flush=True)
                time.sleep(0.0167)
                continue


            # 에피소드 종료 생존시간
            survival_time = time.time() - epi_start_time

            # 타겟 하드 업데이트
            if self.TARGET_UPDATE == TargetUpdate.HARD:
                episode_since_update += 1
                # 타겟 모델 업데이트 후 1000프레임 이상, 3에피소드 이상인 경우 타겟 모델 업데이트
                if frame_since_update > self.UPDATE_FREQ and episode_since_update >= 3:
                    self.target_model.load_state_dict(self.model.state_dict())
                    self.xprint(
                        f"\n타겟 모델 업데이트 after {episode_since_update}episodes, {frame_since_update}frames"
                    )
                    frame_since_update = 0
                    episode_since_update = 0
            # 결과 출력
            self.xprint(
                f"Episode {episode} | Survived {survival_time:.3f} | Max_Q {max_q:.3f} | Total Reward {reward_sum:.2f} | Epsilon {self.epsilon:.2f}"
            )

            # 베스트 모델 저장 & 학습률 감소
            if (episode + 1) % 100 == 0:
                # 사망 시 에이전트의 시야 확인
                frames = next_state.squeeze(0).numpy()
                frames = (frames * 255).astype(np.uint8)
                self.env.vision.record_death(frames, episode)
                # 실제 테스트로 모델 검증
                self.xprint(f"{episode + 1}에피소드 완료. 모델 테스트 5회 진행 중")
                self.validate_model(num_test=5)
                # 학습률 감소
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = max(1e-5, param_group["lr"] * 0.95)
                self.xprint(f"학습률 감소 : {self.optimizer.param_groups[0]['lr']:.7f}")
            elif (episode > half_episode) and (survival_time > self.best_score * 1.2):
                self.xprint(f"{episode + 1}에피소드에서 기존 기록 20% 이상 갱신, 모델 테스트 5회 진행 중")
                self.validate_model(num_test=5)

            # 판이 끝날 때마다 점차 무작위 탐험(epsilon) 확률을 0.5%씩 줄여나감(최저값은 0.05)
            self.epsilon = max(self.EPSILON_MIN, self.epsilon * self.EPSILON_DECAY)

            # 텐서보드 로그
            _add_scalar("Performance/1_Survival_Time", survival_time, episode)
            _add_scalar("Performance/2_Total_Reward", reward_sum, episode)
            _add_scalar("Brain/Max Q-Value", max_q, episode)
            if self.DQN_Type == DQNType.DUELING:
                _add_scalar("Brain/Average Value", sum_value / epi_frame_cnt, episode)
                _add_scalar("Brain/Average Advantage", sum_advantage / epi_frame_cnt, episode)
            self.writer.flush()

        # 훈련 종료
        self.writer.close()
    
    def xprint(self, text, end="\n", flush=False):
        if self.LOGGING:
            print(text, end=end, flush=flush)


if __name__ == "__main__":
    import argparse
    # 1. 인자 parser 생성
    parser = argparse.ArgumentParser(description="Chrome Dino RL Agent")

    # 2. 실행 시 받을 옵션 정의
    parser.add_argument("--dqn_type", type=int, choices=[0, 1, 2, 3], help="DQN 타입(0:Vanilla, 1:Nature, 2:Double, 3:Dueling)")
    parser.add_argument("--img_process", type=int, choices=[0, 1, 2], help="이미지 전처리 방식(0:흑백만, 1:윤곽선검출, 2:프레임차이(difference))")
    parser.add_argument("--buffer_type", type=int, choices=[0, 1, 2], help="경험 기억용 버퍼 타입(0:단일버퍼, 1:우선도+노멀 듀얼버퍼, 2:하이브리드 듀얼 버퍼)")
    parser.add_argument("--target_update", type=int, choices=[0, 1], help="타겟 네트워크 업데이트 타입(0:1000프레임마다 하드 업데이트, 1:소프트 업데이트)")
    parser.add_argument("--num_episodes", type=int, help="훈련 반복 수")
    parser.add_argument("--without_log", action="store_true", help="로그 및 이미지 출력 여부")
    
    # 3. 입력값 파싱
    args = parser.parse_args()
    
    # 4. 입력받은 옵션을 dictionary에 담기
    custom_kwargs = {k: v for k, v in vars(args).items() if v is not None}
    # 5. 하이퍼 파라미터 객체 생성
    hp = HyperParameters(**custom_kwargs)
    # 6. 에이전트 훈련 인스턴스 생성
    trainer = TrainAgent(hp=hp)
    # 에이전트 훈련
    trainer.train_agent()