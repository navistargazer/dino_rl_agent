import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # 1. 학습 가능한 파라미터 (평균 mu, 분산 sigma)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # 2. 노이즈 텐서 (학습되는 값이 아니므로 register_buffer로 등록)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        # 초기화 실행
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """논문에 나온 방식대로 mu와 sigma를 초기화합니다."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

    def _scale_noise(self, size):
        """Factorized Gaussian 노이즈를 생성하는 핵심 수학 공식입니다."""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """새로운 노이즈를 샘플링합니다. (매 스텝 혹은 에피소드마다 호출해야 함)"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # 외적(outer product)을 통해 in x out 크기의 노이즈 행렬 생성 (Factorized 방식)
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        """순전파: 학습 중일 때만 노이즈를 더하고, 평가 중(eval)일 때는 mu만 사용"""
        if self.training: 
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)