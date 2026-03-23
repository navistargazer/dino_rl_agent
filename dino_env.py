from get_state import GetState
import actions as act
import numpy as np
import torch
import time

class DinoEnvironment:
    def __init__(self):
        self.get_state = GetState()
        self.state = self.get_state.get_next_state(isfirst=True)
        self.reward = 0
        self.done = False
        act.click((self.get_state.monitor['left'], self.get_state.monitor['top']))

    def restart_game(self):
        while self.get_state.isgameover:
            act.jump()
            act.wait()
            time.sleep(0.02)
            self.get_state.capture()

        self.state = self.get_state.get_next_state(isfirst=True)
        self.reward = 0
        self.done = False
        return self.state, self.reward, self.done

    def select_action(self, model, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(3)
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
        
        # chrome 렌더링 대기 시간
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
    