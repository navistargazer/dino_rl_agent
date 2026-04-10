from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time


class SeleniumAction:
    def __init__(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--disable-infobars")
        options.add_argument("--window-position=0,0")
        options.add_argument("--window-size=800,500")

        # 크롬의 모든 SSL/인증서 보안 경고창을 강제로 무시합니다.
        options.add_argument("--ignore-certificate-errors")
        options.add_argument("--ignore-ssl-errors")

        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
        )
        self.open_dino_game()

        self.current_key = None

    def open_dino_game(self):
        # CDP로 완벽한 오프라인 상태 만들기
        self.driver.execute_cdp_cmd("Network.enable", {})
        self.driver.execute_cdp_cmd(
            "Network.emulateNetworkConditions",
            {
                "offline": True,
                "latency": 0,
                "downloadThroughput": 0,
                "uploadThroughput": 0,
            },
        )

        # URL: 구글이나 localhost 대신 '존재하지 않는 로컬 IP'를 찌릅니다.
        # 이렇게 하면 크롬이 보안(https) 검사를 아예 시도조차 하지 않고 순수하게 '접속 거부' 에러를 냅니다.
        try:
            self.driver.get("http://127.0.0.1:65432")
        except:
            pass

        time.sleep(1)

        # 아래부터는 아까와 동일한 CSS 암살 & 스페이스바 콤보입니다!
        safe_css = """
        var style = document.createElement('style');
        style.innerHTML = `
            #main-message { display: none !important; }
            #buttons { display: none !important; }
            .error-code { display: none !important; }
        `;
        document.head.appendChild(style);
        """
        self.driver.execute_script(safe_css)

        # self.inject_key("keyDown", 32)
        # time.sleep(0.5)
        # self.inject_key("keyUp", 32)

    def inject_key(self, event="keyDown", key_code=38):
        self.driver.execute_cdp_cmd(
            "Input.dispatchKeyEvent", {"type": event, "windowsVirtualKeyCode": key_code}
        )

    # 대기상태(아무것도 안함 + 이전 행동의 취소)
    def wait(self):
        if not self.current_key:
            return

        if self.current_key == 38:
            # pag.keyUp('up')
            self.inject_key("keyUp", 38)
        elif self.current_key == 40:
            # pag.keyUp('down')
            self.inject_key("keyUp", 40)
        self.current_key = None

    # 점프 키를 누름(떼는 것은 학습에 의해 판단)
    def jump(self):
        # 이전에 숙이기였다면 해제하고
        if self.current_key == 40:
            # pag.keyUp('down')
            self.inject_key("keyUp", 40)
        # 점프 중이 아닐 때만 점프
        if self.current_key != 38:
            # pag.keyDown('up')
            self.inject_key("keyDown", 38)
            self.current_key = 38

    # 숙이기 키를 누름(떼는 것은 학습에 의해 판단)
    def duck(self):
        # 이전에 점프였다면 해제하고
        if self.current_key == 38:
            # pag.keyUp('up')
            self.inject_key("keyUp", 38)
        # 숙이기 중이 아닐 때만
        if self.current_key != 40:
            # pag.keyDown('down')
            self.inject_key("keyDown", 40)
            self.current_key = 40

    def click(self, coord):
        # pag.click(coord)
        pass
