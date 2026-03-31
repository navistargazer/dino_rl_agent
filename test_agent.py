"""
chrome의 dino 게임을 플레이하는 인공지능 (테스트용)
순수한 모델 성능 테스트 스크립트 (학습, 엡실론-그리디 미적용)
"""

import numpy as np
import time
import torch
import os
import csv
from environments import DQN, D3QN, Environment

from dataclasses import dataclass
from enum import IntEnum


# 하이퍼 파라미터 열거형 정의
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


@dataclass
class HyperParameters:
    dqn_type: DQNType = DQNType.DUELING
    img_process: ImgProcess = ImgProcess.DIFF
    buffer_type: BufferType = BufferType.HYBRID
    pixel: int = 64
    num_episodes: int = 50
    fps: int = 15
    without_log: bool = False

    def __post_init__(self):
        self.dqn_type = DQNType(self.dqn_type)
        self.img_process = ImgProcess(self.img_process)
        self.buffer_type = BufferType(self.buffer_type)


class TestAgent:
    def __init__(self, hp: HyperParameters = HyperParameters()):
        self.DQN_Type = hp.dqn_type
        self.IMG_PROCESS_TYPE = hp.img_process
        self.BUFFER_TYPE = hp.buffer_type
        self.PIXEL = hp.pixel
        self.NUM_EPISODES = hp.num_episodes
        self.FPS = hp.fps
        self.LOGGING = not hp.without_log

        print(
            f"[INFO] Test Agent - DQN:{self.DQN_Type.name}, Buffer:{self.BUFFER_TYPE.name}"
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.xprint(f"사용 디바이스: {self.device}")

        # online model
        CNN = DQN if self.DQN_Type < DQNType.DUELING else D3QN
        self.model = CNN(num_actions=3, input_pixel=self.PIXEL).to(self.device)
        self.env = Environment(self.IMG_PROCESS_TYPE, self.PIXEL, logging=self.LOGGING)
        self._restart_game = self.env.restart_game
        self._step = self.env.step
        self.frame_time = 1.0 / self.FPS

        # 모델 로드
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.model_name = f"best_model_{self.DQN_Type.name}_{self.BUFFER_TYPE.name}_{self.IMG_PROCESS_TYPE.name}_{self.TARGET_UPDATE.name}"
        self.model_path = os.path.join(BASE_DIR, f"models/{self.model_name}.pth")

        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)  # 예외 처리
            self.xprint(f"{self.model_name} 모델 로드 완료.")
        else:
            self.xprint(
                f"[경고] {self.model_path} 파일이 없습니다. 처음부터 무작위 가중치로 진행합니다."
            )

        # 평가 모드 설정
        self.model.eval()

    def run_test(self):
        self.xprint(
            f"\n{self.NUM_EPISODES} 에피소드 동안 순수 모델 성능 테스트를 시작합니다."
        )

        survival_records = []
        _argmax = torch.argmax

        # 미분 및 기울기 계산 무시 (역전파 방지)
        with torch.no_grad():
            for episode in range(self.NUM_EPISODES):
                state, done = self._restart_game()
                epi_start_time = time.time()
                epi_frame_cnt = 0

                while not done:
                    start = time.time()
                    epi_frame_cnt += 1

                    q_values = self.model(state.to(self.device))
                    # 엡실론 없이 100% 무조건 최대 보상(Q값) 행동 수행 (Greedy Policy)
                    act_idx = _argmax(q_values).item()

                    if epi_frame_cnt % 15 == 0:
                        self.xprint(".", end="", flush=True)

                    next_state, reward, done = self._step(act_idx)

                    interval = time.time() - start
                    if interval < self.frame_time:
                        time.sleep(self.frame_time - interval)

                    state = next_state

                survival_time = time.time() - epi_start_time
                survival_records.append(survival_time)
                self.xprint(
                    f"\nEpisode {episode + 1}/{self.NUM_EPISODES} | Survived {survival_time:.3f} sec"
                )

        # 통계 계산
        mean_val = np.mean(survival_records)
        std_val = np.std(survival_records)
        max_val = np.max(survival_records)
        median_val = np.median(survival_records)

        self.xprint("\n" + "=" * 40)
        self.xprint(f"[{self.model_name}] 테스트 결과 요약")
        self.xprint(f"평균 생존 기간: {mean_val:.3f} 초")
        self.xprint(f"표준 편차    : {std_val:.3f} 초")
        self.xprint(f"최대 생존 기간: {max_val:.3f} 초")
        self.xprint(f"중앙값       : {median_val:.3f} 초")
        self.xprint("=" * 40)

        # CSV 저장
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        result_dir = os.path.join(BASE_DIR, "results")
        os.makedirs(result_dir, exist_ok=True)
        csv_filename = os.path.join(result_dir, f"test_stats_{self.model_name}.csv")

        with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # 통계 데이터 기록
            writer.writerow(["Metric", "Value (Seconds)"])
            writer.writerow(["Mean", f"{mean_val:.3f}"])
            writer.writerow(["Std", f"{std_val:.3f}"])
            writer.writerow(["Max", f"{max_val:.3f}"])
            writer.writerow(["Median", f"{median_val:.3f}"])
            writer.writerow([])
            # 에피소드 전체 기록 포함
            writer.writerow(["Episode", "Survival Time (Seconds)"])
            for i, record in enumerate(survival_records):
                writer.writerow([i + 1, f"{record:.3f}"])

        self.xprint(f"\n결과가 CSV 파일로 저장되었습니다: {csv_filename}")

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
    test_agent = TestAgent(hp=hp)
    test_agent.run_test()
