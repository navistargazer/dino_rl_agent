import cv2
import numpy as np
import mss
import torch

# 💡 OS/해상도/템플릿 크기를 무시하는 궁극의 다중 비율 매칭 마법
def get_monitor_settings_all_os(template_path='template.png'):
    with mss.mss() as sct:
        # 1. 모니터 전체 화면 캡처 (논리적 해상도 반환)
        # ⭐️ 윈도우의 경우 디스플레이 배율(125%, 150%) 설정이 켜져있더라도 이 좌표는 스케일링되지 않은 논리 좌표입니다.
        monitor = sct.monitors[1]
        
        # 실제 캡처한 이미지(full_screen)는 물리적(Physical) 해상도입니다. (레티나 등은 2배일 수 있음)
        sct_img = sct.grab(monitor)
        full_screen = np.array(sct_img)
        full_screen_gray = cv2.cvtColor(full_screen, cv2.COLOR_BGRA2GRAY)
        
        # 2. 템플릿 이미지 로드
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print(f"❌ '{template_path}' 파일을 찾을 수 없습니다.")
            return None
            
        tH, tW = template.shape[:2]
        
        # ⭐️ 3. 다중 비율 매칭: 템플릿 크기를 0.5배 ~ 1.5배까지 20단계로 나눠서 스캔!
        found = None # 최고의 매칭 결과를 담을 보따리
        
        # np.linspace(0.5, 1.5, 20)은 [0.5, 0.552, ..., 1.5] 처럼 20개의 숫자를 만듭니다.
        scales = np.linspace(0.5, 1.5, 20)
        
        print(f"🕵️‍♂️ 크기를 {scales[0]}~{scales[-1]}배까지 바꾸며 공룡을 스캔합니다. (20회 연산)")
        
        for scale in scales:
            # 템플릿을 현재 비율로 resizing
            resized_template_w = int(tW * scale)
            resized_template_h = int(tH * scale)
            
            # 너무 작거나 크면 패스 (OpenCV 에러 방지)
            if resized_template_w < 10 or resized_template_h < 10 or resized_template_w > full_screen_gray.shape[1] or resized_template_h > full_screen_gray.shape[0]:
                continue
                
            resized_template = cv2.resize(template, (resized_template_w, resized_template_h), interpolation=cv2.INTER_AREA)
            
            # 매칭 실행
            result = cv2.matchTemplate(full_screen_gray, resized_template, cv2.TM_CCOEFF_NORMED)
            (_, max_val, _, max_loc) = cv2.minMaxLoc(result)
            
            # 만약 지금까지 찾은 인식률보다 더 높다면 갱신!
            if found is None or max_val > found[0]:
                found = (max_val, max_loc, scale) # (인식률, 위치, 비율) 저장

        # ⭐️ 최고의 결과 확인
        (max_val, max_loc, scale) = found
        
        print(f"✅ 최고의 인식률 발견: {max_val:.2f} (최적 비율: {scale:.2f}배)")
        
        if max_val < 0.85: # 인식률이 0.85 이하면 믿을 수 없음
            print("❌ 공룡을 찾지 못했습니다. 게임창을 띄우고 다시 실행해 주세요.")
            return None
            
        # 4. 물리적 좌표(px, py)를 논리적 좌표(lx, ly)로 변환 (레티나 대응)
        # full_screen_gray가 물리적 해상도이므로 배율을 계산합니다.
        # ⭐️ 윈도우 100% 배율 모니터에서는 이 값이 1.0이 됩니다.
        physical_h, physical_w = full_screen_gray.shape
        scale_factor_x = physical_w / monitor['width']
        scale_factor_y = physical_h / monitor['height']
        
        lx = int(max_loc[0] / scale_factor_x)
        ly = int(max_loc[1] / scale_factor_y)
        
        # 5. 찾은 공룡을 기준으로 게임 영역(MONITOR) 계산
        # 질문자님이 원하시는 게임창의 논리적 크기 (가로 600, 세로 200)
        game_logical_w = 350
        game_logical_h = 82
        
        monitor_settings = {
            'top': ly - 30,  
            'left': lx - 260,
            'width': game_logical_w,
            'height': game_logical_h
        }
        
        print("\n✅ 궁극의 MONITOR 좌표를 찾았습니다!")
        print(f"MONITOR = {monitor_settings}")
        
        # 6. 시각적 확인 (팝업창)
        check_img = sct.grab(monitor_settings)
        cv2.imshow("Check 캡처 영역 (Q를 누르면 종료)", cv2.cvtColor(np.array(check_img), cv2.COLOR_BGRA2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return monitor_settings

if __name__ == "__main__":
    get_monitor_settings_all_os()