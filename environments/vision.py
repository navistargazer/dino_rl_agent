import cv2
import numpy as np
import torch
import mss
from collections import deque
import os

class Vision:
    def __init__(self, img_process_type=2, pixel=64, logging=True):
        self.IMG_PROCESS_TYPE = img_process_type
        self.PIXEL = pixel
        print(f'[INFO] 이미지 전처리 타입: {self.IMG_PROCESS_TYPE}, 이미지 크기: {self.PIXEL}x{self.PIXEL}')
        self.sct = mss.mss()
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(cur_dir, 'template.png')
        self.monitor = self.find_monitor(path, logging)
        self.frames_stacked = deque(maxlen=4)
        self.isgameover = False
        self.prev_frame = None

    def find_monitor(self, template_path, logging=True):
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
        
        if logging:
            # 7. 시각적 검증 (디버그용)
            check_img = self.sct.grab(monitor_settings)
            cv2.imshow("Capture(Press any key to continue)", cv2.cvtColor(np.array(check_img), cv2.COLOR_BGRA2BGR))
            cv2.waitKey(0) # 임의의 키 입력 시 대기 종료
            cv2.destroyAllWindows()
        
        return monitor_settings

    # 화면 프레임 캡처 - 게임오버도 판단
    def grab_monitor(self):
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
        return gray

    def get_processed_image(self):
        curr_frame = self.grab_monitor()
        # 이미지 전처리(0:none, 1:canny, 2:diff)
        if self.IMG_PROCESS_TYPE == 0:
            # 0: 전처리 없음
            processed = curr_frame
        elif self.IMG_PROCESS_TYPE == 1:
            # 1:윤곽선만 따서 다크모드 대응
            processed = cv2.Canny(curr_frame, 100, 200)
        elif self.IMG_PROCESS_TYPE == 2:
            # 2:이전 프레임과 차이가 있는 부분만 밝게, 나머지는 0
            # 게임 시작/재시작시에는 초기화 후 장 채움(또는 이전 프레임이 없는 경우)
            if self.prev_frame is None:
                # 첫번째 프레임은 움직임이 없으므로 전부 0(검은 화면)
                self.prev_frame = curr_frame
                diff = cv2.absdiff(curr_frame, curr_frame)
            else:
                # 현재 프레임에서 이전 프레임을 빼고, 이전 프레임으로 입력
                diff = cv2.absdiff(curr_frame, self.prev_frame)
                self.prev_frame = curr_frame
            # diff(프레임 차이)에서 확실한 차이(50차이 이상)만 흰색으로
            # 구름, 달과 배경의 차이 = 37, 장애물과 배경 차이는 172
            _, processed = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
        else:
            processed = curr_frame
        # 리사이즈 및 노멀라이즈
        resized = cv2.resize(processed, (self.PIXEL, self.PIXEL), interpolation=cv2.INTER_AREA)
        normalized = (resized / 255.0).astype(np.float32)
        return normalized



    # 게임 진행용 함수 (매 프레임마다 호출)
    # 캡처 화면 4장을 전처리 후 스태킹해서 텐서 형태로 리턴
    def get_next_state(self, isfirst=False):
        normalized = self.get_processed_image()
        if isfirst:
            # 처음엔 최초 diff=0으로 4장을 채움
            self.frames_stacked.clear()
            self.frames_stacked.extend([normalized] * 4)
        else:
            # 다음 스테이트 용으로 프레임 추가
            self.frames_stacked.append(normalized)

        state = torch.from_numpy(np.stack(self.frames_stacked, axis=0)).unsqueeze(0)
        return state

if __name__ == "__main__":
    # 1. Vision 객체 생성
    vision = Vision()
    
    import time
    print("크롬 공룡 게임 창을 클릭해서 활성화해 주세요! (10초 뒤 캡처 시작)")
    time.sleep(10)
    
    # 원본 흑백 프레임 4장을 모아둘 큐 (테스트 시각화용)
    original_frames_q = deque(maxlen=4)
    
    # 2. 첫 프레임 초기화
    state_tensor = vision.get_next_state(isfirst=True)
    original_frames_q.append(vision.prev_frame) # 첫 원본 프레임 저장
    
    # 3. 공룡이 실제로 달리는 궤적을 얻기 위해 프레임 진행
    print("달리는 모션을 캡처하는 중입니다...")
    for _ in range(15):
        time.sleep(1.0 / 15.0) # 게임 프레임(15 FPS) 속도에 맞춰 대기
        state_tensor = vision.get_next_state(isfirst=False)
        original_frames_q.append(vision.prev_frame) # 매 프레임 원본 흑백 저장
        
    # --- [이미지 1: 원본 흑백 화면 세로로 4장 스택] ---
    # original_frames_q에는 과거(위) -> 현재(아래) 순서로 4장의 원본 이미지가 들어있음
    gray_stack = cv2.vconcat(list(original_frames_q))
    
    # --- [이미지 2: 공룡의 시야 (Diff) 세로로 4장 스택] ---
    # 텐서에서 이미지를 꺼내어 0~255 값으로 복원
    frames = state_tensor.squeeze(0).numpy()
    frames = (frames * 255).astype(np.uint8)
    # 과거(위) -> 현재(아래) 순서로 세로 스택
    diff_stack = cv2.vconcat([frames[0], frames[1], frames[2], frames[3]])
    
    # ---------------------------
    
    # 파일로 각각 저장
    save_gray_path = "gray_original_4stack.png"
    save_diff_path = "ai_vision_4stack.png"
    cv2.imwrite(save_gray_path, gray_stack)
    cv2.imwrite(save_diff_path, diff_stack)
    
    print(f"캡처 완료! '{save_gray_path}'와 '{save_diff_path}' 파일이 저장되었습니다.")
    print("창을 닫거나 아무 키나 누르면 프로그램이 종료됩니다.")
    
    # 화면에 창 2개 띄우기 (확대 없이 원본 크기 그대로)
    cv2.imshow("1) Original Gray (4 Stack)", gray_stack)
    cv2.imshow("2) AI Vision Diff (4 Stack)", diff_stack)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()