import pyautogui
import time
from dqn_vision import Vision

class DinoEnvironment:
    def __init__(self, monitor):
        self.vision = Vision(monitor)
        self.state = self.vision.get_next_state(isfirst=True)
        self.reward = 0
        self.done = False

    def restart_game(self):
        pyautogui.hotkey('alt', 'tab')
        time.sleep(1)
        pyautogui.press('space')
        self.state = self.vision.get_next_state(isfirst=True)
        self.reward = 0
        self.done = False
        return self.state, self.reward, self.done

    def step(self, action):
        # action = 0(아무것도 안함)이라면 1프레임 기다림
        if action == 0:
            self.wait()
        # action = 1(점프)라면 점프의 체공시간만큼 기다림
        elif action == 1:
            self.jump()
        # action = 2(숙이기)라면 짧은 시간 기다림
        else:
            self.down()
        # 행동 이후 상태
        self.state = self.vision.get_next_state()
        # 사망 판정
        self.done = self._is_game_over()
        # 보상 설정
        if self.done:
            self.reward = -10
        else:
            self.reward = 0.1 if action == 0 else 0.0

        return self.state, self.reward, self.done

    def _is_game_over(self):
        # GAMEOVER 특정 픽셀 값을 받아와서 검게 변했으면 사망 판정
        '''=== 클릭한 픽셀 정보 ====
Matplotlib 좌표 (x, y): (47.86, 2.55)
넘파이 배열 인덱스 [y, x]: [2, 47]
픽셀 값 (0~1 정규화): 0.3255
👉 수정할 코드: state[3, 2, 47]'''
        dead_pixel = self.state

    def wait(self):
        print('wait')
    
    def jump(self):
        pyautogui.keyDown('up')
        time.sleep(0.4)
        pyautogui.keyUp('up')
        print('jump')

    def down(self):
        pyautogui.keyDown('down')
        pyautogui.keyUp('down')
        print('down')
    