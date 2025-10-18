from collections import deque
import time
import pyautogui

from screenshots import take_screenshot
from agent import Agent

N = 4
frame_height, frame_width = 84, 84

frame_stack = deque(maxlen=N)

x, y = 964, 78
width, height = 952, 458

agent = Agent()

pressed = {'left': False, 'right': False, 'a': False, 's': False}

def hold_key(key):
    if not pressed[key]:
        pyautogui.keyDown(key)
        pressed[key] = True

def release_key(key):
    if pressed[key]:
        pyautogui.keyUp(key)
        pressed[key] = False

while True:
    frame = take_screenshot(x, y, width, height)
    frame_stack.append(frame)
    if len(frame_stack) == N:
        throttle, steering = agent.select_action(frame_stack)

        # throttle
        if throttle > 0:
            hold_key('s'); release_key('a')
        elif throttle < 0:
            hold_key('a'); release_key('s')
        else:
            release_key('a'); release_key('s')

        # steering
        if steering > 0:
            hold_key('right'); release_key('left')
        elif steering < 0:
            hold_key('left'); release_key('right')
        else:
            release_key('left'); release_key('right')

    time.sleep(0.1)