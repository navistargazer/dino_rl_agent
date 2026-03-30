from .vision import Vision
from .action import Action
import time
import platform


class Environment:
    def __init__(self, img_process_type=2, pixel=64, logging=True):
        self.is_not_mac = platform.system() != "Darwin"
        self.vision = Vision(img_process_type, pixel, logging=logging)
        self.action = Action()
        self.state = self.vision.get_next_state(isfirst=True)
        self.reward = 0
        self.coord = (self.vision.monitor["left"], self.vision.monitor["top"])

    @property
    def is_game_over(self):
        return self.vision.gameover_detected

    def restart_game(self):
        start = time.time()
        self.action.click(self.coord)
        while not self.is_game_over:
            self.action.jump()
            self.action.wait()
            if self.is_not_mac:
                time.sleep(0.02)
            self.vision.grab_monitor()

            if time.time() - start > 2:
                break
        self.action.click((self.vision.monitor["left"], self.vision.monitor["top"] + 200))
        self.action.release_all()
        for _ in range(5):
            self.state = self.vision.get_next_state()
            time.sleep(0.2)
        self.reward = 0
        return self.state, self.is_game_over

    def step(self, action):
        # action = 0(아무것도 안함)이라면 1프레임 기다림
        if action == 0:
            self.action.wait()
        # action = 1(점프)라면 점프의 체공시간만큼 기다림
        elif action == 1:
            self.action.jump()
        # action = 2(숙이기)라면 짧은 시간 기다림
        else:
            self.action.duck()

        # chrome 렌더링 대기 시간
        if self.is_not_mac:
            time.sleep(0.0167)

        # 행동 이후 상태
        self.state = self.vision.get_next_state()
        # 보상 설정
        self.reward = -1 if self.is_game_over else 0.001
        return self.state, self.reward, self.is_game_over


