import torch
import torch.nn as nn
import torch.nn.functional as F
from .noisy_linear import NoisyLinear

class LayerParams:
    def __init__(self, kernel_size, stride, padding):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

class D3QN(nn.Module):
    def __init__(self, input_shape=(4, 64, 256), num_actions=3, noisy_net=True):
        super(D3QN, self).__init__()
        layer_1 = LayerParams(5, 4, 2)
        layer_2 = LayerParams(3, 2, 1)
        layer_3 = LayerParams(3, 2, 1)
        # 합성곱 계층, 입력:(배치, 채널, 행, 열) = (1, 4, 64, 256)
        self.conv1 = nn.Conv2d(
            input_shape[0], 32, kernel_size=layer_1.kernel_size, stride=layer_1.stride, padding=layer_1.padding
        )  # (1, 32, 16, 64)
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=layer_2.kernel_size, stride=layer_2.stride, padding=layer_2.padding
        )  # (1, 64, 8, 32)
        self.conv3 = nn.Conv2d(
            64, 64, kernel_size=layer_3.kernel_size, stride=layer_3.stride, padding=layer_3.padding
        )  # (1, 64, 4, 16)

        # FC 층(Q-Value 계산)
        # 256x64 이미지가 세 번의 conv를 거치면 4x16 크기가 됨 (4096 = 64 * 4 * 16)
        # 크기 계산용 더미 텐서를 사용
        dummy_input = torch.empty(1, *input_shape)
        with torch.no_grad():
            dummy_output = F.relu(self.conv1(dummy_input))
            dummy_output = F.relu(self.conv2(dummy_output))
            dummy_output = F.relu(self.conv3(dummy_output))
        flatten_size = dummy_output.view(1, -1).size(1)  # (1, 4096)

        self.noisy_net = noisy_net

        # Dueling Network
        # noisy nets 도입
        if self.noisy_net:
            # 1. 가치(Value) 흐름 - value of state
            self.value_fc = NoisyLinear(flatten_size + 1, 512)
            self.value = NoisyLinear(512, 1)
            # 2. 행동(Advantage) 흐름 - advantage of action
            self.adv_fc = NoisyLinear(flatten_size + 1, 512)
            self.advantage = NoisyLinear(512, num_actions)
        else:
            # 1. 가치(Value) 흐름 - value of state
            self.value_fc = nn.Linear(flatten_size + 1, 512)  # (1, 512)
            self.value = nn.Linear(512, 1)  # (1, 1)
            # 2. 행동(Advantage) 흐름 - advantage of action
            self.adv_fc = nn.Linear(flatten_size + 1, 512)  # (2, 512)
            self.advantage = nn.Linear(512, num_actions)  # (1, 3)

        # (선택) 가중치 초기화: ReLU를 사용하는 네트워크의 국룰
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity="relu")
        if not self.noisy_net:
            nn.init.kaiming_uniform_(self.value_fc.weight, nonlinearity="relu")
            nn.init.kaiming_uniform_(self.adv_fc.weight, nonlinearity="relu")

    def forward(self, state, return_dueling=False):
        # 상태를 이미지스택과 시간으로 언패킹
        x, t = state

        # 합성곱-활성화 3번
        x = F.relu(self.conv1(x))  # (1, 32, 16, 16)
        x = F.relu(self.conv2(x))  # (1, 64, 8, 8)
        x = F.relu(self.conv3(x))  # (1, 64, 4, 4)
        # 데이터를 1차원으로 flatten
        x = x.view(x.size(0), -1)  # (1, 1024)
        x = torch.cat((x, t), dim=1)  # (1, 1025)

        # FC 층
        # 1. value 계산
        val = F.relu(self.value_fc(x))
        value = self.value(val)

        # 2. advantage 계산
        adv = F.relu(self.adv_fc(x))
        advantage = self.advantage(adv)

        # 3. dueling 결합
        # Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        if return_dueling:
            return q_values, value, advantage
        return q_values

    def reset_noise(self):
            if self.noisy_net:
                self.value_fc.reset_noise()
                self.value.reset_noise()
                self.adv_fc.reset_noise()
                self.advantage.reset_noise()
