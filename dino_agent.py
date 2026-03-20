'''
ai가 키보드를 눌러 공룡을 점프시키거나, 아무 것도 하지 않는 행동
'''
import pyautogui
import numpy as np
import torch
import time

from dqn_cnn import DQN_CNN


class DinoAgent:
    def __init__(self, monitor, epsilon):
        self.brain = DQN_CNN()
        self.epsilon = epsilon

    # 2. 현재 상태에 맞는 동작(action) 실행 - 동작:3(0-아무것도 안함, 1-점프, 2-숙이기)
    def select_action(self, state):
        # epsilon보다 작은 경우는 랜덤 행동, 크면 q-value에 의한 행동
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1, 2])
        else:
            with torch.no_grad():               # 가중치 저장 안함(미분x, 역전파x)
                q_values = self.brain.forward(state) # 현재 상태의 q값
                return torch.argmax(q_values).item()    # 가장 큰 값의 인덱스에 있는 숫자(int)를 반환 -> q값이 [0.1, 0.8]인 경우 1번 인덱스의 item인 1을 반환
    
    def jump(self):
        pyautogui.keyDown('up')
        time.sleep(0.2)
        pyautogui.keyUp('up')

    def down(self):
        pyautogui.keyDown('down')
        pyautogui.keyUp('down')
