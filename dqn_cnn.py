import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN_CNN(nn.Module):
    def __init__(self, input_channels=4, num_actions=2):
        super(DQN_CNN,self).__init__()
        # 합성곱 계층, 입력:(1, 4, 84, 84)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)   # (1, 32, 20, 20)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)               # (1, 64, 9, 9)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)               # (1, 64, 7, 7)

        # FC 층(Q-Value 계산)
        # 84x84 이미지가 세 번의 conv를 거치면 7x7 크기가 됨 (3136 = 64 * 7 * 7)
        self.fc1 = nn.Linear(3136, 512)                                         # (1, 512)
        self.fc2 = nn.Linear(512, num_actions)                                  # (1, 2)

    def forward(self, state):
        # 합성곱-활성화 3번
        state = F.relu(self.conv1(state))       # (1, 32, 20, 20)
        statex = F.relu(self.conv2(state))       # (1, 64, 9, 9)
        state = F.relu(self.conv3(state))       # (1, 64, 7, 7)
        # 데이터를 1차원으로 flatten
        state = statex.view(state.size(0), -1)       # (1, 3136)
        # FC 층
        state = F.relu(self.fc1(state))         # (512)
        # Q-Value 계산(행동별 점수)
        q_values = self.fc2(state)          # (2)
        return q_values