import pyautogui as pag

class Action():
    def __init__(self):
        pag.PAUSE = 0.0
        pag.FAILSAFE = False
        self.current_key = None
        pass

    def click(self, coord):
        pag.click(coord)

    # 대기상태(아무것도 안함 + 이전 행동의 취소)
    def wait(self):
        if not self.current_key:
            return

        if self.current_key == 'up':
            pag.keyUp('up')
        elif self.current_key == 'down':
            pag.keyUp('down')
        self.current_key = None

    # 점프 키를 누름(떼는 것은 학습에 의해 판단)
    def jump(self):
        # 이전에 숙이기였다면 해제하고
        if self.current_key == 'down':
            pag.keyUp('down')
        # 점프 중이 아닐 때만 점프
        if self.current_key != 'up':
            pag.keyDown('up')
            self.current_key = 'up'

    # 숙이기 키를 누름(떼는 것은 학습에 의해 판단)
    def duck(self):
        # 이전에 점프였다면 해제하고
        if self.current_key == 'up':
            pag.keyUp('up')
        # 숙이기 중이 아닐 때만
        if self.current_key != 'down':
            pag.keyDown('down')
            self.current_key = 'down'

    def release_all(self):
        if self.current_key:
            pag.keyUp(self.current_key)
            self.current_key = None