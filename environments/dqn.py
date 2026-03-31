import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_channels=4, num_actions=3, input_pixel=64):
        super(DQN, self).__init__()
        # 합성곱 계층, 입력:(1, 4, 64, 64)
        self.conv1 = nn.Conv2d(
            input_channels, 32, kernel_size=7, stride=4, padding=2
        )  # (1, 32, 16, 16)
        pixel = self.get_pixel_size(input_pixel, kernel_size=7, stride=4, padding=2)
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, stride=2, padding=1
        )  # (1, 64, 8, 8)
        pixel = self.get_pixel_size(pixel, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(
            64, 64, kernel_size=3, stride=2, padding=1
        )  # (1, 64, 4, 4)
        pixel = self.get_pixel_size(pixel, kernel_size=3, stride=2, padding=1)

        # FC 층(Q-Value 계산)
        # 64x64 이미지가 세 번의 conv를 거치면 4x4 크기가 됨 (1024 = 64 * 4 * 4)
        flatten_size = 64 * pixel * pixel
        self.fc1 = nn.Linear(flatten_size, 512)  # (1, 512)
        self.fc2 = nn.Linear(512, num_actions)  # (1, 3)

        # (선택) 가중치 초기화: ReLU를 사용하는 네트워크의 국룰
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")

    def get_pixel_size(self, input_size, kernel_size, stride, padding):
        pixel = (input_size - kernel_size + 2 * padding) / stride + 1
        return int(pixel)

    def forward(self, state):
        # 상태를 이미지스택과 시간으로 분리
        x, t = state
        # 합성곱-활성화 3번
        x = F.relu(self.conv1(x))  # (1, 32, 16, 16)
        x = F.relu(self.conv2(x))  # (1, 64, 8, 8)
        x = F.relu(self.conv3(x))  # (1, 64, 4, 4)
        # 데이터를 1차원으로 flatten
        x = x.view(x.size(0), -1)  # (1, 1024)
        # 시간을 텐서에 붙여줌
        x = torch.cat((x, t), dim=1)  # (1, 1025)
        # FC 층
        x = F.relu(self.fc1(x))  # (512)
        # Q-Value 계산(행동별 점수)
        q_values = self.fc2(x)  # (3)
        return q_values
