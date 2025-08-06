import pyautogui
import time

time.sleep(2)  # 2 秒緩衝時間，讓你切到目標程式


while True:
    pyautogui.keyDown('ctrl')  # 模擬按住 Ctrl 鍵

    pyautogui.keyUp('ctrl')

