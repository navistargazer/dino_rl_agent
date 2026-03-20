import cv2
import numpy as np
import torch
import mss
from collections import deque

class Vision:
    def __init__(self, monitor):
        self.monitor = monitor
        self.sct = mss.mss()
        self.frames_stacked = deque(maxlen=4)

    def capture(self):
        screen = np.array(self.sct.grab(self.monitor))
        # 수정 1: BGRA -> GRAY로 정확히 변환
        gray = cv2.cvtColor(screen, cv2.COLOR_BGRA2GRAY) 
        resized = cv2.resize(gray, (84, 84))
        normalized = (resized / 255.0).astype(np.float32)
        return normalized

    # 추가/수정 3: 게임 진행용 함수 (매 프레임마다 호출)
    def get_next_state(self, isfirst=False):
        # 현재 프레임 생성
        frame = self.capture()
        # 게임 시작/재시작시에는 초기화 후 장 채움
        if isfirst:
            self.frames_stacked.clear()
            self.frames_stacked.extend([frame] * 4)
        else:
            # 다음 스테이트 용으로 프레임 추가
            self.frames_stacked.append(frame)
        state = torch.from_numpy(np.stack(self.frames_stacked, axis=0)).unsqueeze(0)
        return state