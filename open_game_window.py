import webbrowser
import pygetwindow as gw
import time

def setup_game_window():
    print("🌐 크롬 공룡 게임을 자동으로 엽니다...")
    
    # # 1. 오프라인 상태를 흉내내는 공룡 게임 클론 사이트 열기
    # # (chrome://dino 는 파이썬에서 보안상 직접 열기 까다로워서 클론 사이트를 씁니다)
    # webbrowser.open("https://chromedino.com/")
    
    # # 브라우저가 뜨고 페이지가 로딩될 때까지 3초 정도 넉넉히 기다려줍니다.
    # time.sleep(3) 
    
    # 2. 열려있는 윈도우 창들 중에서 이름에 'Dino'가 들어간 창을 찾습니다.
    # (chromedino.com 접속 시 창 제목이 보통 "Chrome Dino" 등으로 설정됨)
    windows = gw.getWindowsWithTitle("공룡")
    
    if len(windows) > 0:
        win = windows[0]
        
        # 3. 창 제어 시작!
        if win.isMinimized or win.isMaximized:
            win.restore() # 최소화되어 있다면 원래대로
            
        win.activate() # 창을 맨 앞으로 가져와서 포커스 맞추기
        
        # ⭐️ 여기가 핵심! 창을 모니터 좌측 상단 끝으로 밀어버리고 크기를 고정합니다.
        win.moveTo(0, 0) 
        win.resizeTo(800, 600) 
        
        print("✅ 게임 창 세팅 완료! 위치(0, 0) / 크기(800x600)")
        return True
    else:
        print("❌ 게임 창을 찾을 수 없습니다. 수동으로 세팅해 주세요.")
        return False
        