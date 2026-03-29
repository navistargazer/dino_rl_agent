from .vision import Vision
from . import actions as act
import time
import platform


class Environment:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.is_not_mac = platform.system() != "Darwin"
        self.vision = Vision()
        self.state = self.vision.get_next_state(isfirst=True)
        self.reward = 0
        self.done = False
        self.coord = (self.vision.monitor["left"], self.vision.monitor["top"])

    def restart_game(self):
        start = time.time()
        act.click(self.coord)
        while self.vision.isgameover:
            act.jump()
            act.wait()
            if self.is_not_mac:
                time.sleep(0.02)
            self.vision.capture()

            if time.time() - start > 2:
                break
        act.release_all()
        self.state = self.vision.get_next_state(isfirst=True)
        self.reward = 0
        self.done = False
        return self.state, self.done

    def get_q_values(self, return_dueling=False):
        state_dev = self.state.to(self.device)
        if return_dueling:
            q_values, value, advantage = self.model(state_dev, return_dueling)
            return q_values, value, advantage
        q_values = self.model(state_dev)
        return q_values

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

        # chrome 렌더링 대기 시간
        if self.is_not_mac:
            time.sleep(0.02)

        # 행동 이후 상태
        self.state = self.vision.get_next_state()
        # 사망 판정
        self.done = self.vision.isgameover
        # 보상 설정
        if self.done:
            self.reward = -10
        else:
            if action == 0:
                self.reward = 0.1
            elif action == 1:
                self.reward = 0.0
            else:
                self.reward = 0.05
        return self.state, self.reward, self.done
