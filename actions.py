import pyautogui

pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = False
current_key = None


def click(coord):
    pyautogui.click(coord)

# 대기상태(아무것도 안함 + 이전 행동의 취소)
def wait():
    global current_key
    if not current_key:
        return

    if current_key == 'up':
        pyautogui.keyUp('up')
    elif current_key == 'down':
        pyautogui.keyUp('down')
    current_key = None

# 점프 키를 누름(떼는 것은 학습에 의해 판단)
def jump():
    global current_key
    # 이전에 숙이기였다면 해제하고
    if current_key == 'down':
        pyautogui.keyUp('down')
    # 점프 중이 아닐 때만 점프
    if current_key != 'up':
        pyautogui.keyDown('up')
        current_key = 'up'

# 숙이기 키를 누름(떼는 것은 학습에 의해 판단)
def down():
    global current_key
    # 이전에 점프였다면 해제하고
    if current_key == 'up':
        pyautogui.keyUp('up')
    # 숙이기 중이 아닐 때만
    if current_key != 'down':
        pyautogui.keyDown('down')
        current_key = 'down'

def release_all():
    global current_key
    if current_key:
        pyautogui.keyUp(current_key)
        current_key = None