import cv2
import numpy as np
import torch
import mss
from collections import deque
import os


class GetState:
    def __init__(self):
        self.sct = mss.mss()
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(cur_dir, 'template.png')
        self.monitor = self.find_monitor(path)
        self.frames_stacked = deque(maxlen=4)
        self.isgameover = False

    def find_monitor(self, template_path):
        """
        주어진 템플릿 이미지를 화면에서 검색하여 타겟 게임 영역의 논리적 좌표를 반환합니다.
        다중 스케일 템플릿 매칭을 사용하여 HiDPI 및 다양한 UI 배율 환경에 대응합니다.
        """
        # 기본 모니터(1번) 설정. 다중 모니터 환경 시 인덱스 변경 필요 가능성 존재.
        monitor = self.sct.monitors[1]
        
        # 1. 전체 화면 캡처 및 전처리 (물리 해상도 기준)
        sct_img = self.sct.grab(monitor)
        full_screen = np.array(sct_img)
        full_screen_gray = cv2.cvtColor(full_screen, cv2.COLOR_BGRA2GRAY)
        
        # 2. 템플릿 이미지 로드
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(f"[ERROR] 템플릿 이미지를 로드할 수 없습니다: '{template_path}'")
            
        tH, tW = template.shape[:2]
        
        # 3. 다중 스케일 템플릿 매칭 (0.5x ~ 1.5x, 20 steps)
        best_match = None 
        scales = np.linspace(0.5, 1.5, 20)
        
        print(f"[INFO] 템플릿 매칭 시작 (스케일 범위: {scales[0]:.2f}x ~ {scales[-1]:.2f}x, {len(scales)}단계)")
        
        for scale in scales:
            resized_w = int(tW * scale)
            resized_h = int(tH * scale)
            
            # 리사이즈된 템플릿이 유효하지 않거나 원본 화면보다 큰 경우 예외 처리
            if (resized_w < 10 or resized_h < 10 or 
                resized_w > full_screen_gray.shape[1] or 
                resized_h > full_screen_gray.shape[0]):
                continue
                
            resized_template = cv2.resize(template, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
            
            # CCOEFF_NORMED 매칭 수행 (1.0에 가까울수록 일치율 높음)
            result = cv2.matchTemplate(full_screen_gray, resized_template, cv2.TM_CCOEFF_NORMED)
            (_, max_val, _, max_loc) = cv2.minMaxLoc(result)
            
            # 최고 신뢰도(Confidence) 갱신
            if best_match is None or max_val > best_match[0]:
                best_match = (max_val, max_loc, scale)

        # 4. 매칭 결과 검증
        if best_match is None:
            raise RuntimeError("[ERROR] 유효한 템플릿 매칭을 수행하지 못했습니다.")
            
        (max_val, max_loc, best_scale) = best_match
        print(f"[INFO] 최적 매칭 결과 - 신뢰도: {max_val:.4f}, 최적 스케일: {best_scale:.2f}x")
        
        # 신뢰도 임계값(Threshold) 검사 (오탐지 방지)
        THRESHOLD = 0.85
        if max_val < THRESHOLD:
            raise RuntimeError(f"[ERROR] 템플릿 매칭 실패 (신뢰도 {max_val:.4f} < {THRESHOLD}). 화면에 대상이 노출되어 있는지 확인하십시오.")
            
        # 5. 물리적 좌표를 논리적 좌표로 변환 (HiDPI 보정)
        physical_h, physical_w = full_screen_gray.shape
        scale_factor_x = physical_w / monitor['width']
        scale_factor_y = physical_h / monitor['height']
        
        logic_x = int(max_loc[0] / scale_factor_x)
        logic_y = int(max_loc[1] / scale_factor_y)
        
        # 6. 타겟 게임 영역 좌표 산출
        OFFSET_X = -230
        OFFSET_Y = -30
        GAME_WIDTH = 384
        GAME_HEIGHT = 84
        
        monitor_settings = {
            'top': logic_y + OFFSET_Y,  
            'left': logic_x + OFFSET_X,
            'width': GAME_WIDTH,
            'height': GAME_HEIGHT
        }
        
        print(f"[INFO] 산출된 캡처 영역 좌표: {monitor_settings}")
        
        # 7. 시각적 검증 (디버그용)
        check_img = self.sct.grab(monitor_settings)
        cv2.imshow("Capture(Press any key to continue)", cv2.cvtColor(np.array(check_img), cv2.COLOR_BGRA2BGR))
        cv2.waitKey(0) # 임의의 키 입력 시 대기 종료
        cv2.destroyAllWindows()
        
        return monitor_settings

    # 화면 프레임 캡처 - 게임오버도 판단
    def capture(self):
        screen = np.array(self.sct.grab(self.monitor))
        # 수정 1: BGRA -> GRAY로 정확히 변환
        gray = cv2.cvtColor(screen, cv2.COLOR_BGRA2GRAY) 
        # gameover : 배경과 글씨 픽셀의 차이로 판단(낮과 밤 동시 적용됨)
        # self.isgameover = abs(int(gray[0, 0]) - int(gray[0, 185])) > 100.0
        # 맥과 윈도우의 픽셀 밀림을 영역으로 판단함으로써 해결
        bg_pixel = int(gray[0, 100])
        gameover = gray[0:3, 200:205].astype(int)
        diff_area = np.abs(gameover - bg_pixel)
        self.isgameover = np.max(diff_area) > 100
        # 밤/낮 상관없이 윤곽선으로 학습하기 위한 윤곽선 이미지
        edge = cv2.Canny(gray, 100, 200)
        # 84x84로 리사이즈(이미지 축소 시 윤곽선이 날아가지 않기 위해 면적평균 보간)
        resized = cv2.resize(edge, (64, 64), interpolation=cv2.INTER_AREA)
        normalized = (resized / 255.0).astype(np.float32)
        return normalized

    # 게임 진행용 함수 (매 프레임마다 호출)
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

if __name__ == "__main__":
    state = GetState()