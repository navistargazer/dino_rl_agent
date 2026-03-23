import pyautogui

pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = False

def click(coord):
    pyautogui.click(coord)

# 대기상태(아무것도 안함 + 이전 행동의 취소)
def wait():
    pyautogui.keyUp('up')
    pyautogui.keyUp('down')

# 점프 키를 누름(떼는 것은 학습에 의해 판단)
def jump():
    pyautogui.keyUp('down') # 이전에 숙이기였다면 해제하고
    pyautogui.keyDown('up')

# 숙이기 키를 누름(떼는 것은 학습에 의해 판단)
def down():
    pyautogui.keyUp('up')   # 이전에 점프였다면 해제하고
    pyautogui.keyDown('down')