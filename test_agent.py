"""
chrome의 dino 게임을 플레이하는 인공지능 (실전 테스트 및 실험 데이터 수집용)
훈련된 모델의 성능을 50회 측정하고, 결과를 CSV로 저장하여 박스플롯 비교에 사용합니다.
"""

import time
import torch
import numpy as np
import csv
import os
import config as cf
from environments import DQN, D3QN, Environment


def test_agent():
    # 1. 초기화 (환경, 모델 준비)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Mac GPU (MPS) 가동!")
    else:
        device = torch.device("cpu")
        print("CPU 가동!")

    _DQN_VER = cf.DQN_VER
    CNN = DQN if _DQN_VER < 3 else D3QN
    model = CNN(num_actions=3, input_pixel=cf.PIXEL).to(device)
    env = Environment(model)
    frame_time = 1.0 / cf.FPS

    # 2. 학습된 베스트 모델 로드
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_name = f"best_model_DQN{cf.DQN_VER}_{cf.NUM_BUFFER}Buffer"
    model_path = os.path.join(BASE_DIR, f"models/{model_name}.pth")

    print(f"\n[INFO] {model_path} 로드 시도...")

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✅ 학습된 뇌 이식 완료! (실험 대상: {model_name})\n")
    else:
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        print("경고: 학습되지 않은 깡통 뇌(무작위 가중치)로 플레이합니다.\n")

    model.eval()

    # ⭐️ 3. 실험 설정 및 기록 보따리 준비
    _TEST_EPISODES = 50  # 박스플롯을 위한 통계적 유의미한 횟수 (50회)
    survival_records = []
    reward_records = []

    _restart_game = env.restart_game
    _step = env.step
    _argmax = torch.argmax
    _time = time.time
    _sleep = time.sleep

    print(f"🚀 [{model_name}] 실전 성능 테스트 {_TEST_EPISODES}회 가동을 시작합니다!")

    with torch.no_grad():
        for episode in range(_TEST_EPISODES):
            state, done = _restart_game()

            epi_start_time = _time()
            epi_frame_cnt = 1
            reward_sum = 0.0

            while not done:
                start = _time()

                # 테스트 중에는 진행 상황만 간단히 출력 (로그 화면 도배 방지)
                if epi_frame_cnt % 30 == 0:
                    print(".", end="", flush=True)

                q_values = model(state.to(device))
                action = _argmax(q_values).item()

                epi_frame_cnt += 1
                next_state, reward, done = _step(action)

                interval = _time() - start
                if interval < frame_time:
                    _sleep(frame_time - interval)

                state = next_state
                reward_sum += reward

            # ⭐️ 에피소드 종료 후 기록 저장
            survival_time = _time() - epi_start_time
            survival_records.append(survival_time)
            reward_records.append(reward_sum)

            print(
                f" [Ep {episode + 1:02d}] 생존: {survival_time:.2f}초 | 보상: {reward_sum:.2f}"
            )

    # ⭐️ 4. 실험 결과 통계 출력 및 CSV 저장
    avg_survival = np.mean(survival_records)
    std_survival = np.std(survival_records)
    max_survival = np.max(survival_records)

    print("\n" + "=" * 50)
    print(f"🎯 [{model_name}] 테스트 50회 결과 요약")
    print(f" - 평균 생존 시간 : {avg_survival:.2f} 초 (±{std_survival:.2f})")
    print(f" - 최고 생존 시간 : {max_survival:.2f} 초")
    print("=" * 50)

    # 결과를 CSV 파일로 저장 (나중에 박스플롯 그릴 때 사용)
    result_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(result_dir, exist_ok=True)
    csv_filename = os.path.join(result_dir, f"test_results_{model_name}.csv")
    # csv_filename = results/test_results_{model_name}.csv"

    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "SurvivalTime", "TotalReward"])  # 헤더(Header) 작성
        for i in range(_TEST_EPISODES):
            writer.writerow([i + 1, survival_records[i], reward_records[i]])

    print(f"💾 실험 데이터가 저장되었습니다: {csv_filename}")


if __name__ == "__main__":
    test_agent()
