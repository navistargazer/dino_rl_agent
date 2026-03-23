import pyautogui
import time

def click(coord):
    pyautogui.click(coord)


def wait():
        pass
    
def jump():
    pyautogui.keyDown('up')
    time.sleep(0.4)
    pyautogui.keyUp('up')

def down():
    pyautogui.keyDown('down')
    pyautogui.keyUp('down')