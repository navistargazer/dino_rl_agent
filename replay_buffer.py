from collections import deque
import random



class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    # 버퍼 메모리에 tuple 저장
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # batch 수 만큼 랜덤 샘플링
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    # 현재 버퍼에 쌓인 수 리턴 len(replaybuffer) 의 값
    def __len__(self):
        return len(self.buffer)