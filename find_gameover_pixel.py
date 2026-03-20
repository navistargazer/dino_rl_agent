# find_game_over_coords.py
import cv2
import numpy as np
import mss
import matplotlib.pyplot as plt

# --- 사용자의 monitor 설정과 동일하게 맞추세요 ---
# dql_vision.py에 설정된 monitor 값을 그대로 가져옵니다.
monitor = {'top': 170, 'left': 180, 'width': 600, 'height': 100}
sct = mss.mss()

def capture_and_preprocess():
    """게임을 플레이하다가 죽었을 때 호출: 화면을 캡처하고 84x84 흑백으로 변환"""
    print("\n--- [도구] Game Over 화면 캡처 시작 ---")
    print("1. 크롬 공룡 게임을 켭니다.")
    print("2. 일부러 선인장에 박아서 'GAME OVER' 화면을 띄웁니다.")
    print("3. 이 터미널에서 Enter 키를 누르면 화면을 캡처합니다.")
    input("Press Enter when ready...")

    # 화면 캡처 및 전처리 (우리의 비전 모듈과 동일 로직)
    screen = np.array(sct.grab(monitor))
    gray = cv2.cvtColor(screen, cv2.COLOR_BGRA2GRAY)
    resized = cv2.resize(gray, (84, 84))
    
    # 정규화 (0~1) - 실제 에이전트가 보는 값과 똑같이 만듭니다.
    normalized = (resized / 255.0).astype(np.float32)
    
    return normalized

def onclick(event):
    """Matplotlib 차트 위를 클릭했을 때 호출되는 콜백 함수"""
    if event.xdata is not None and event.ydata is not None:
        x_click, y_click = event.xdata, event.ydata
        
        # 소수점 좌표를 넘파이 배열 인덱스용 정수로 변환
        x_int, y_int = int(x_click), int(y_click)
        
        print("\n=== 클릭한 픽셀 정보 ====")
        print(f"Matplotlib 좌표 (x, y): ({x_click:.2f}, {y_click:.2f})")
        
        # ★★★ 매우 중요 ★★★
        # Matplotlib 플롯의 (x, y)는 넘파이 배열의 [y, x] (행, 열) 인덱스가 됩니다!
        print(f"넘파이 배열 인덱스 [y, x]: [{y_int}, {x_int}]")
        
        # 해당 픽셀의 값 (0~1 정규화된 값)
        pixel_val = image[y_int, x_int]
        print(f"픽셀 값 (0~1 정규화): {pixel_val:.4f}")
        
        # 실제 적용할 코드 가이드
        print(f"👉 수정할 코드: state[3, {y_int}, {x_int}]")
        print("=========================")
        
        # 한 번 클릭하면 창을 닫도록 설정
        plt.close()

if __name__ == "__main__":
    # 1. 죽어있는 화면 캡처
    image = capture_and_preprocess()

    # 2. Matplotlib으로 이미지 띄우기
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray', origin='upper')
    ax.set_title("Click on a dark pixel of Game Over UI\n(e.g., inside 'R' in RESTART)")
    
    # 마우스 클릭 이벤트 연결
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    # 눈금(Grid)을 켜면 좌표를 어림잡기 좋습니다.
    ax.grid(True, which='both', color='red', linestyle='-', linewidth=0.5)
    
    print("\n👉 띄워진 이미지 창에서 'GAME OVER' 글씨나 리스타트 버튼의 ")
    print("   '검은색 부분'을 마우스로 한 번 클릭하세요.")
    plt.show()