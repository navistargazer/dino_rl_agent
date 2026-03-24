import cv2
import numpy as np
from mss import mss
import config as cf

def init_recorder(monitor):
    # ⭐️ [녹화용 추가] 비디오 라이터 세팅 (루프 밖에서 한 번만 설정)
    sct = mss()
    monitor = env.get_state.monitor_settings # 캡처 영역 가져오기
    fps = 1.0 / cf.FRAME_INTERVAL
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

def name_episode():
    # ⭐️ [녹화용 추가] 에피소드마다 파일명 다르게 생성
    video_filename = f'recordings/play_ep{episode}.avi'
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (monitor['width'], monitor['height']))

def capture_monitor():
     # ⭐️ [녹화용 추가] 화면 캡처 및 영상 프레임 추가
    sct_img = sct.grab(monitor)
    frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR) # BGRA를 BGR로 변환
    video_writer.write(frame)

def save_video():
     # ⭐️ [녹화용 추가] 에피소드가 끝나면 영상 파일 저장 완료하기
    video_writer.release()
    print(f"🎬 {video_filename} 저장 완료!")