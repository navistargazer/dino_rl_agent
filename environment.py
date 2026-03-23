from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import base64
import cv2
import numpy as np

class DinoEnvironment:
    def __init__(self):
        # 크롬 옵션 설정
        chrome_options = Options()
        chrome_options.add_argument("--mute-audio")
        # chrome_options.add_argument("--headless")

        # 브라우저 실행 및 웹사이트 접속
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        self.driver.get("chrome://dino")
        
        # 브라우저 크기 고정
        self.driver.set_window_size(800, 600)
        time.sleep(2)

        # 게임을 감싸고 있는 캔버스 요소 찾기
        self.canvas = self.driver.find_element("class name", "runner-canvas")

    def restart_game(self):
        # 게임 재시작
        self.driver.execute_script("Runner.instance_.restart()")
        time.sleep(1)
        self.state = self.get_state.get_next_state(isfirst=True)
        self.reward = 0
        self.done = False
        return self.state, self.reward, self.done

    def select_action(self, model, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice([0, 1])
        else:
            with torch.no_grad():
                q_values = model.forward(self.state)
                return torch.argmax(q_values).item()
        
    def step(self, action):
        # action = 0(아무것도 안함)이라면 1프레임 기다림
        if action == 0:
            act.wait()
        # action = 1(점프)라면 점프의 체공시간만큼 기다림
        elif action == 1:
            act.jump()
        # action = 2(숙이기)라면 짧은 시간 기다림
        else:
            act.down()
        # 행동 이후 상태
        self.state = self.get_state.get_next_state()
        # 사망 판정
        self.done = self.get_state.isgameover
        # 보상 설정
        if self.done:
            self.reward = -10
        else:
            self.reward = 0.1 if action == 0 else 0.0
        return self.state, self.reward, self.done
    
    def get_state(self):
        # 캔버스(게임 화면) 영역만 Base64 텍스트 형태로 스크린샷 캡처
        image_b64 = self.canvas.screenshot_as_base64
        
        # Base64 텍스트를 OpenCV 이미지(Numpy Array)로 변환
        image_bytes = base64.b64decode(image_b64)
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_GRAYSCALE)
        
        # 여기서 기존처럼 공룡 코앞부터 자르고 84x84로 Resize 하는 로직 수행!
        # img = img[위:아래, 공룡_코앞_X위치:우측끝] 
        # img = cv2.resize(img, (84, 84))
        
        return img
    
    def is_game_over(self):
        # 게임 내부 JS 변수에 접근해서 죽었는지(True/False) 직접 물어봄
        is_crashed = self.driver.execute_script("return Runner.instance_.crashed;")
        return is_crashed
    
if __name__ == "__main__":
    env = DinoEnvironment()
    env.restart_game()