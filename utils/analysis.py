import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def plot_survival_boxplots(results_dir="results", save_path="results/survival_boxplot.png"):
    """
    results 폴더 내의 test_stats_*.csv 파일들을 읽어들여 
    에피소드별 생존 시간의 박스 플롯을 그립니다. 여러 모델의 결과를 비교하기 좋습니다.
    """
    # utils 폴더의 부모 디렉토리 (프로젝트 루트 경로)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    results_path = os.path.join(base_dir, results_dir)
    
    # results 폴더 안의 모든 테스트 통계 csv 탐색
    csv_files = glob.glob(os.path.join(results_path, "test_stats_*.csv"))
    
    if not csv_files:
        print(f"[경고] {results_path} 경로에서 해당하는 CSV 파일을 찾을 수 없습니다.")
        print("hint: test.py를 먼저 실행하여 평가를 기록해주세요.")
        return
        
    data = []
    labels = []
    
    for file in csv_files:
        filename = os.path.basename(file)
        # 파일명에서 'test_stats_best_model_' 부분과 '.csv'를 제거해 모델의 특징적 이름만 추출
        model_name = filename.replace("test_stats_best_model_", "").replace("test_stats_", "").replace(".csv", "")
        
        # 앞의 6줄(통계 요약 데이터)을 건너뛰고 7번째 줄을 헤더로 하여 데이터프레임 로드
        try:
            df = pd.read_csv(file, skiprows=6)
            
            # 생존 시간 열 이름 찾기
            survival_col = [col for col in df.columns if "Survival" in col]
            if not survival_col:
                print(f"[경고] {filename} 파일 안에 Survival 열이 없습니다. 건너뜁니다.")
                continue
                
            survival_col = survival_col[0]
            
            # 결측치 제거 후 배열화
            survival_times = df[survival_col].dropna().values
            data.append(survival_times)
            labels.append(model_name)
        except Exception as e:
            print(f"[오류] {filename} 파일을 읽는 중 문제가 발생했습니다: {e}")
            
    if not data:
        print("[경고] 박스 플롯을 그릴 유효한 데이터가 없습니다.")
        return
        
    # 창 크기 및 디자인 세팅
    plt.figure(figsize=(10, 6))
    
    # 박스 플롯 그리기 (showmeans 옵션을 켜서 평균값을 마커로 표시)
    plt.boxplot(data, labels=labels, showmeans=True)
    
    # 제목 및 축 설정
    plt.title("Model Survival Time Comparison", fontsize=16)
    plt.ylabel("Survival Time (Seconds)", fontsize=14)
    plt.xlabel("Tested Models", fontsize=14)
    
    # 라벨 이름이 길 수 있으므로 45도 기울임 처리
    plt.xticks(rotation=45, ha='right')
    
    # y축 점선 그리드
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 여백 자동 조정 (라벨 겹치지 않게)
    plt.tight_layout()
    
    # 플롯 이미지로 최종 저장
    save_file = os.path.join(base_dir, save_path)
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    plt.savefig(save_file)
    print(f"✅ 박스 플롯 비교 이미지가 성공적으로 저장되었습니다: {save_file}")
    
    # 화면 팝업으로도 띄워주기
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CSV 결과를 바탕으로 박스플롯 비교 그래프 그리기")
    parser.add_argument("--save_path", type=str, default="results/survival_boxplot.png", help="박스 플롯을 저장할 경로")
    
    args = parser.parse_args()
    plot_survival_boxplots(save_path=args.save_path)
