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

        return self.state, self.reward, self.done

    def _is_game_over(self):
        pass

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
    