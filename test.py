from pynput.keyboard import Key,Controller
import time

keyboard = Controller()
keyboard.press(Key.media_volume_up)
keyboard.release(Key.media_volume_up)
time.sleep(0.1)