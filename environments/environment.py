from .vision import Vision
from .action import Action
from .selenium_action import SeleniumAction
import time
import platform


class Environment:
    def __init__(
        self, img_process_type=1, pixel=64, logging=True, rewards=(-1, 0.01)
    ):
        self.is_not_mac = platform.system() != "Darwin"
        self.action = SeleniumAction()
        self.vision = Vision(img_process_type, pixel, logging=logging)
        self.reward_done, self.reward_alive = rewards
        self.tick = 0
        self.state = self.vision.get_next_state(isfirst=True)
        self.coord = (self.vision.monitor["left"] + 400, self.vision.monitor["top"])

    @property
    def is_game_over(self):
        return self.vision.gameover_detected

    def restart_game(self):
        self.tick = 0
        self.action.jump()
        while self.is_game_over:
            self.action.jump()
            time.sleep(0.0167)
            self.action.wait()
            self.vision.grab_monitor()

        for _ in range(5):
            self.state = self.vision.get_next_state()
            self.action.click(self.coord)
            time.sleep(0.2)
        state = (self.state, self.tick)
        return state, self.is_game_over

    def step(self, action):
        reward = self.reward_alive
        # action = 0(아무것도 안함)이라면 1프레임 기다림
        if action == 0:
            self.action.wait()
        # action = 1(점프)라면 점프의 체공시간만큼 기다림
        elif action == 1:
            # reward *= 0.5
            self.action.jump()
        # action = 2(숙이기)라면 짧은 시간 기다림
        else:
            # reward *= 0.8
            self.action.duck()
        # chrome 렌더링 대기 시간
        if self.is_not_mac:
            time.sleep(0.0167)

        # 행동 이후 상태
        self.state = self.vision.get_next_state()
        # 상태에 시간 변수를 추가(노멀라이즈 개념으로 작은 값을 더해줌)
        self.tick += 1e-4
        state = (self.state, self.tick)

        # 사망 보상 설정
        if self.is_game_over:
            reward = self.reward_done

        return state, reward, self.is_game_over
