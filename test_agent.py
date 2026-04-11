"""
chrome의 dino 게임을 플레이하는 인공지능 (테스트용)
순수한 모델 성능 테스트 스크립트 (학습, 엡실론-그리디 미적용)
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

class Exploration(IntEnum):
    NONE = 0
    EPSILON_GREEDY = 1
    NOISY_NET = 2



@dataclass
class HyperParameters:
    # DQN 버전 (0:vanilla, 1:nature, 2:double, 3:dueling )
    dqn_type: DQNType = DQNType.DUELING
    # 화면 이미지 전처리 방식 (0:흑백만, 1:윤곽선검출, 2:프레임차이(difference)) 
    img_process: ImgProcess = ImgProcess.CANNY
    # 기억용 버퍼 타입 (0:단일버퍼, 1:우선도+노멀 듀얼버퍼, 2:하이브리드 듀얼 버퍼)
    buffer_type: BufferType = BufferType.HYBRID
    # 타겟 네트워크 업데이트 타입 (0:1000프레임마다 하드 업데이트, 1:소프트 업데이트)
    target_update: TargetUpdate = TargetUpdate.HARD
    # 탐험 정책(0:엡실론-그리디, 1:노이지 네트워크)
    exploration: Exploration = Exploration.NOISY_NET
    # CNN 인풋용 리사이즈 픽셀 세로 크기
    pixel: int = 64
    # 행동 보상 (사망, 대기)
    rewards: tuple[float, float] = (-1, 0.01)
    # 훈련 반복 수
    num_episodes: int = 1000
    # 미니 배치 훈련에 사용할 과거 경험 개수
    batch_size: int = 32
    # 우선도 버퍼에서 꺼내는 비율
    p_ratio: float = 0.5
    # 우선도 버퍼 크기
    p_buffer_size: int = 20000
    # 노멀 버퍼 크기
    n_buffer_size: int = 100000
    # 초당 프레임 수
    fps: int = 15
    # 엡실론(랜덤 탐험 비율) 최소값
    epsilon_min: float = 0.01
    # 에피소드 당 엡실론 감소율
    epsilon_decay: float = 0.995
    # 타겟 네트워크 업데이트 주기
    update_freq: int = 1000
    # 타겟 네트워크 소프트 업데이트 비율
    tau: float = 0.002
    # 학습률
    learning_rate: float = 0.0001
    # 미래가치 할인율
    gamma: float = 0.97
    # 로그 및 확인 이미지 출력 여부
    without_log: bool = False

    # 들어온 값을 enum 오브젝트로 캐스팅
    def __post_init__(self):
        self.dqn_type = DQNType(self.dqn_type)
        self.img_process = ImgProcess(self.img_process)
        self.buffer_type = BufferType(self.buffer_type)
        self.target_update = TargetUpdate(self.target_update)


class TestAgent:
    def __init__(self, hp: HyperParameters = HyperParameters(), logging=True):
        self.DQN_Type = hp.dqn_type
        self.IMG_PROCESS_TYPE = hp.img_process
        self.BUFFER_TYPE = hp.buffer_type
        self.TARGET_UPDATE = hp.target_update
        self.Exploration = hp.exploration
        self.PIXEL = hp.pixel
        self.REWARDS = hp.rewards
        self.NUM_EPISODES = hp.num_episodes
        self.BATCH_SIZE = hp.batch_size
        self.P_RATIO = hp.p_ratio
        self.P_BUFFER_SIZE = hp.p_buffer_size
        self.N_BUFFER_SIZE = hp.n_buffer_size
        self.FPS = hp.fps
        self.EPSILON_MIN = hp.epsilon_min
        self.EPSILON_DECAY = hp.epsilon_decay
        self.UPDATE_FREQ = hp.update_freq
        self.TAU = hp.tau
        self.LEARNING_RATE = hp.learning_rate
        self.GAMMA = hp.gamma
        self.LOGGING = not hp.without_log

        # 하이퍼 파라미터 출력
        print(
            f"[INFO] DQN:{self.DQN_Type.name} | buffer:{self.BUFFER_TYPE.name} | ImgProcess:{self.IMG_PROCESS_TYPE.name} | TargetUpdate:{self.TARGET_UPDATE.name} | Exploration:{self.Exploration.name}", 
            f"\n[INFO] REWARD:[사망, 생존]={self.REWARDS} | FPS:{self.FPS} | LR:{self.LEARNING_RATE} | GAMMA:{self.GAMMA} | TAU:{self.TAU}"
        )

        # 디바이스 설정
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.xprint(f"사용 디바이스: {self.device}")

        # 강화학습에서는 pytorch의 과도한 멀티스레드에 의한 cpu오버헤드 방지
        # torch.set_num_threads(1)

        # online model - 버전에 따라 DQN(vanilla, nature, double), D3QN(dueling) 선택
        CNN = DQN if self.DQN_Type < DQNType.DUELING else D3QN
        # 모델에 전달하는 파라미터 딕셔너리 - **로 언패킹해서 쓰면 됨
        model_params = {
            "input_shape": (4, self.PIXEL, self.PIXEL * 4),
            "num_actions": 3,
            "noisy_net": self.Exploration == Exploration.NOISY_NET,
        }
        self.model = CNN(**model_params).to(self.device)  # 학습자의 두뇌
        # target model - nature DQN부터 타겟 네트워크 가동
        # dqn 버전이 올라가면 타겟 모델 등장
        if self.DQN_Type > 0:
            self.target_model = CNN(**model_params).to(self.device)  # 목표 신경망
            # 타겟 모델에 현행 모델의 상태 주입
            self.target_model.load_state_dict(self.model.state_dict())
            # 타겟 모델은 학습 없이 평가모드로
            if self.Exploration != Exploration.NOISY_NET:
                self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        # 키보드 제어, 보상 판단을 통제할 환경 인스턴스
        self.env = Environment(
            self.IMG_PROCESS_TYPE,
            self.PIXEL,
            logging=self.LOGGING,
            rewards=self.REWARDS,
        )
        # 환경 함수 캐싱
        self._restart_game = self.env.restart_game
        self._step = self.env.step

        self.frame_time = 1.0 / self.FPS

        # 기억 저장용 버퍼
        self.replaybuffer = ReplayBuffer(
            priority_cap=self.P_BUFFER_SIZE,
            normal_cap=self.N_BUFFER_SIZE,
            priority_ratio=self.P_RATIO,
        )

        # 이어서 학습할 경우를 위한 변수
        self.best_score = 0
        self.epsilon = 1.0

        # 학습 이어하기(모델 저장 파일 로드)
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.model_name = f"{self.DQN_Type.name}_{self.BUFFER_TYPE.name}_{self.IMG_PROCESS_TYPE.name}_{self.TARGET_UPDATE.name}_{self.Exploration.name}"
        # self.best_model_path = os.path.join(self.BASE_DIR, f"models/{self.model_name}.pth")
        self.best_model_path = "C:\dev\projects\dino_rl_agent\models\DUELING_HYBRID_CANNY_HARD_NOISY_NET.pth"
        self.xprint(f"{self.model_name} 모델 사용,", end=" ")
        if os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.best_score = checkpoint["best_score"]
            self.epsilon = max(checkpoint["epsilon"], 0.1)
            self.xprint(
                f"이어서 학습 시작 (기존 최고 생존: {self.best_score} / Epsilon: {self.epsilon:.3f})"
            )
        else:
            os.makedirs(os.path.join(self.BASE_DIR, "models"), exist_ok=True)
            self.best_score = 0
            self.epsilon = 1.0
            self.xprint("새로운 학습 시작")

        # 그래프 저장 경로
        plot_dir = os.path.join(self.BASE_DIR, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        self.plot_path = os.path.join(plot_dir, f"{self.model_name}.png")
        # tensorboard 로그 경로
        log_dir = os.path.join(self.BASE_DIR, "runs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{self.model_name}")
        self.writer = SummaryWriter(log_path)  # run/self.model_name 폴더에 로그가 쌓임

        self.history_q_values = []
        self.history_td_error = []
        self.history_loss = []
        self.history_survived = []

    def validate_model(self, num_test=50):
        """
        훈련 도중 엡실론을 제거한 진짜 실력을 검증하고 베스트 모델을 저장
        epsilon = 0, 버퍼 저장 및 배치 훈련 안함
        """
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
                # 단일 에피소드 시작
                while not done:
                    start = time.time()
                    # 1. 도델의 최대 Q 값에 의한 행동 결정
                    img, tick = state
                    # img_tensor = img.to(self.device).float() / 255.0
                    img_tensor = (
                        torch.from_numpy(img).unsqueeze(0).to(self.device).float()
                        / 255.0
                    )
                    tick_tensor = torch.tensor(
                        tick, dtype=torch.float32, device=self.device
                    ).view(1, 1)
                    state = (img_tensor, tick_tensor)
                    q_values = self.model(state)
                    act_idx = _argmax(q_values).item()

                    # 2. 다음 상태로 진행
                    next_state, reward, done = self._step(act_idx)
                    # 6. 프레임 간격보다 짧은 시간에 끝났다면 기다림
                    interval = time.time() - start
                    if interval < self.frame_time:
                        time.sleep(self.frame_time - interval)
                    elif interval > self.frame_time * 3:
                        self.xprint(
                            f"frame delayed: {interval - self.frame_time:.3f}sec"
                        )
                    if done:
                        max_q = _max(q_values).item()
                    reward_sum += reward
                    state = next_state

                # 에피소드 종료 생존시간
                survival_time = time.time() - epi_start_time
                # # 기록
                # self.history_q_values.append(max_q)
                # self.history_survived.append(survival_time)
                # 결과 출력
                self.xprint(
                    f"Test:{i+1}/{num_test} | Survived:{survival_time:.3f} | Max_Q:{max_q:.3f} | Total Reward:{reward_sum:.2f}"
                )
                survival_record.append(survival_time)

        # 최대 생존 시간
        best_score = np.mean(survival_record).item()
        # 베스트 모델 저장
        if best_score > self.best_score:
            self.best_score = best_score
            self.xprint(f"{best_score:.3f}로 베스트 모델이 갱신되었습니다.")
        else:
            self.xprint("모델 갱신 실패")
        

    def xprint(self, text, end="\n", flush=False):
        if self.LOGGING:
            print(text, end=end, flush=flush)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chrome Dino Pure Test Agent")

    parser.add_argument(
        "--dqn_type",
        type=int,
        choices=[0, 1, 2, 3],
        help="DQN 타입(0:Vanilla, 1:Nature, 2:Double, 3:Dueling)",
    )
    parser.add_argument(
        "--img_process",
        type=int,
        choices=[0, 1, 2],
        help="이미지 전처리 방식(0:흑백만, 1:윤곽선검출, 2:프레임차이(difference))",
    )
    parser.add_argument(
        "--buffer_type",
        type=int,
        choices=[0, 1, 2],
        help="경험 기억용 버퍼 타입(0:단일버퍼, 1:우선도+노멀 듀얼버퍼, 2:하이브리드 듀얼 버퍼)",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=50, help="테스트 반복 수 (기본 50회)"
    )
    parser.add_argument(
        "--without_log", action="store_true", help="로그 및 이미지 출력 여부"
    )

    args = parser.parse_args()
    custom_kwargs = {k: v for k, v in vars(args).items() if v is not None}

    hp = HyperParameters(**custom_kwargs)
    test_agent = TestAgent(
        hp=hp, logging=True
    )
    test_agent.validate_model()
