"""
chrome의 dino 게임을 플레이하는 인공지능 (테스트용)
순수한 모델 성능 테스트 스크립트 (학습, 엡실론-그리디 미적용)
"""

import numpy as np
import time
import torch
from environments import DQN, D3QN, Environment
import os

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
    # DQN 버전 (0:vanilla, 1:nature, 2:double, 3:dueling )
    dqn_type: DQNType = DQNType.DOUBLE
    # 화면 이미지 전처리 방식 (0:흑백만, 1:윤곽선검출, 2:프레임차이(difference))
    img_process: ImgProcess = ImgProcess.CANNY
    # 기억용 버퍼 타입 (0:단일버퍼, 1:우선도+노멀 듀얼버퍼, 2:하이브리드 듀얼 버퍼)
    buffer_type: BufferType = BufferType.HYBRID
    # 타겟 네트워크 업데이트 타입 (0:1000프레임마다 하드 업데이트, 1:소프트 업데이트)
    target_update: TargetUpdate = TargetUpdate.HARD
    # CNN 인풋용 리사이즈 픽셀크기
    pixel: int = 64
    # 행동 보상 (사망, 대기)
    rewards: tuple[float, float] = (-1, 0.01)
   # 초당 프레임 수
    fps: int = 15
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
        self.PIXEL = hp.pixel
        self.REWARDS = hp.rewards
        self.NUM_EPISODES = hp.num_episodes
        self.FPS = hp.fps
        self.LOGGING = not hp.without_log

        # 하이퍼 파라미터 출력
        print(
            f"[INFO] DQN:{self.DQN_Type.name} | buffer:{self.BUFFER_TYPE.name} | ImgProcess:{self.IMG_PROCESS_TYPE.name} | TargetUpdate:{self.TARGET_UPDATE.name} | REWARD:[사망, 생존]={self.REWARDS} | FPS:{self.FPS} | LR:{self.LEARNING_RATE} | GAMMA:{self.GAMMA} | TAU:{self.TAU}"
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
        self.model = CNN(num_actions=3, input_pixel=self.PIXEL).to(
            self.device
        )  # 학습자의 두뇌
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

       # 학습 이어하기(모델 저장 파일 로드)
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.model_name = f"{self.DQN_Type.name}_{self.BUFFER_TYPE.name}_{self.IMG_PROCESS_TYPE.name}_{self.TARGET_UPDATE.name}"
        self.model_path = os.path.join(self.BASE_DIR, f"models/{self.model_name}.pth")
        self.xprint(f"{self.model_name} 모델 사용,", end=" ")
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            # self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.best_score = checkpoint["best_score"]
            self.xprint(
                f"이어서 학습 시작 (기존 최고 생존: {self.best_score} / Epsilon: {self.epsilon:.3f})"
            )
        else:
            os.makedirs(os.path.join(self.BASE_DIR, "models"), exist_ok=True)
            self.best_score = 0
            self.epsilon = 1.0
            self.xprint("새로운 학습 시작")

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
