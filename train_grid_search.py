import subprocess
import itertools
import time

def run_grid_search():
    # 1. 테스트 할 하이퍼파라미터
    dqn_types = [0, 1, 2, 3]
    img_process_types = [0, 1, 2]
    buffer_types = [0, 1, 2]
    target_update_types = [0, 1]
    episodes = 1000
    # 모든 경우의 수 생성(4*3*3*2) = 72
    experiments = list(itertools.product(dqn_types, img_process_types, buffer_types, target_update_types))
    total_runs = len(experiments)
    
    for idx, (dqn, img, buf, target) in enumerate(experiments):
        print(f'{idx+1}/{total_runs} 테스트 진행 중...({dqn}_DQN, {img}_img, {buf}_buf, {target}_update)')

        # 2. 실행할 터미널 명령어
        command = [
            'python', 'train_agent.py',
            '--dqn_type', str(dqn),
            '--img_process', str(img),
            '--buffer_type', str(buf),
            '--target_update', str(target),
            '--num_episodes', str(episodes),
            '--without_log'
        ]

        # 3. subprocess 로 훈련 스크립트 실행
        start_time = time.time()
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f'\n테스트[{idx+1}/{total_runs}] 실패: {e.returncode}')
            continue
        except KeyboardInterrupt:
            print('\n사용자에 의해 테스트 종료됨. 전체 실험을 종료합니다.')
            break
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n테스트[{idx+1}/{total_runs}] 완료! ({elapsed_time:.2f}초 소요)")

        # 휴식 시간
        time.sleep(3)
    print("모든 실험이 완료되었습니다.")

if __name__ == "__main__":
    run_grid_search()
        

