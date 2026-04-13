from collections import deque
import random


class ReplayBuffer:
    def __init__(self, priority_cap=2000, normal_cap=8000, priority_ratio=0.5):
        # 의미있는 기억을 뽑아내는 비율(0:완전랜덤, 1:극단적인 것만)
        self.p_ratio = priority_ratio  
        self.p_buffer = deque(maxlen=priority_cap)
        self.n_buffer = deque(maxlen=normal_cap)

    # 버퍼 메모리에 tuple 저장
    def push_to_normal(self, experience):
        self.n_buffer.append(experience)
    
    def push_to_priority(self, experience):
        self.p_buffer.append(experience)

    # batch 수 만큼 랜덤 샘플링
    def sample(self, batch_size, buffer_type):
    # def sample(self, batch_size, buffer_type, epi_progress):
        # # 우선도 버퍼에서 뽑을 샘플 숫자
        # max_ratio = 1  # 최고 비율
        # # # 에피소드가 진행될수록 우선도 버퍼 비율을 max_ratio에서 p_ratio로 감소
        # p_ratio = max_ratio - (max_ratio - self.p_ratio) * epi_progress
        # # 버퍼 수에 따라 p_buffer 사이즈 결정
        # p_size = int(batch_size * p_ratio) if buffer_type > 0 else 0
        p_size = int(batch_size * self.p_ratio) if buffer_type > 0 else 0
        # 우선도 버퍼에서 뽑을 숫자가 모자라는 경우
        if (p_size > len(self.p_buffer)):
            p_size = len(self.p_buffer)
        # 남은 숫자는 노멀 버퍼에서 뽑음
        n_size = batch_size - p_size

        # 각 버퍼에서 랜덤 추출 후 합치기
        p_samples = random.sample(self.p_buffer, p_size)
        n_samples = random.sample(self.n_buffer, n_size)
        return p_samples + n_samples

    @property
    def p_buffer_size(self):
        return len(self.p_buffer)

    @property
    def n_buffer_size(self):
        return len(self.n_buffer)

    # 현재 버퍼에 쌓인 수 리턴 len(replaybuffer) 의 값
    def __len__(self):
        return len(self.p_buffer) + len(self.n_buffer)
