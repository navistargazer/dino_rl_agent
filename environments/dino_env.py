from .get_state import GetState
from . import actions as act
import torch
import time
import platform


class DinoEnvironment:
    def __init__(self):
        self.is_not_mac = platform.system() != "Darwin"
        self.get_state = GetState()
        self.state = self.get_state.get_next_state(isfirst=True)
        self.reward = 0
        self.done = False
        self.coord = (self.get_state.monitor["left"], self.get_state.monitor["top"])
        act.click(self.coord)

    def restart_game(self):
        start = time.time()
        while self.get_state.isgameover:
            act.jump()
            act.wait()
            if self.is_not_mac:
                time.sleep(0.02)
            self.get_state.capture()

            if time.time() - start > 2:
                break
        act.release_all()
        self.state = self.get_state.get_next_state(isfirst=True)
        self.reward = 0
        self.done = False
        return self.state, self.done

    def get_q_values(self, model):
        with torch.no_grad():
            device = next(model.parameters()).device
            state_dev = self.state.to(device)
            q_values = model(state_dev)
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
        self.state = self.get_state.get_next_state()
        # 사망 판정
        self.done = self.get_state.isgameover
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
