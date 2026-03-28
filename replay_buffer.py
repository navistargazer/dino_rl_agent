from collections import deque
import numpy as np
import random
import config as cf

class ReplayBuffer:
    def __init__(self, priority_cap=2000, normal_cap=8000, priority_ratio=0.5):
        self.p_ratio = priority_ratio    # 의미있는 기억을 뽑아내는 비율(0:완전랜덤, 1:극단적인 것만)
        self.p_buffer = deque(maxlen=priority_cap)
        self.n_buffer = deque(maxlen=normal_cap)


    # 버퍼 메모리에 tuple 저장
    def push(self, memory):
        # memory = (state, action, reward, next_state, done)
        # 단일버퍼이면
        if cf.NUM_BUFFER == 1:
            self.n_buffer.append(memory)
            return
        # 사망했거나, 점프/숙이기 동작을 한 것은 우선도 높은 기억
        if memory[-1] or memory[1] in [1, 2]:
            self.p_buffer.append(memory)
        # 나머지는 평범한 기억으로
        else:
            self.n_buffer.append(memory)

    # batch 수 만큼 랜덤 샘플링
    def sample(self, batch_size):
        # 우선도 버퍼에서 뽑을 샘플 숫자
        p_size = int(batch_size * self.p_ratio) if cf.NUM_BUFFER == 2 else 0
        n_size = batch_size - p_size

        # 우선도 버퍼에서 뽑을 숫자가 모자라는 경우
        p_size = min(p_size, len(self.p_buffer))
        n_size = min(n_size, len(self.n_buffer))

        # 각 버퍼에서 랜덤 추출 후 합치기
        p_samples = random.sample(self.p_buffer, p_size)
        n_samples = random.sample(self.n_buffer, n_size)
        return p_samples + n_samples
    
    # 현재 버퍼에 쌓인 수 리턴 len(replaybuffer) 의 값
    def __len__(self):
        return len(self.p_buffer) + len(self.n_buffer)