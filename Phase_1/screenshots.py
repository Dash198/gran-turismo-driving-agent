"""
Core input mode of the agent.

Takes screenshots of a part of the screen and returns it to the agent to stack them.
"""

import mss
from PIL import Image
import numpy as np

def take_screenshot(x, y, width, height, img_dims = (224, 224)):
    with mss.mss() as sct:
        monitor = {"top": y, "left": x, "width": width, "height": height}
        sct_img = sct.grab(monitor)

        img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
        img = img.convert("RGB")
        img = img.resize(img_dims)
        preprocessed_img = np.array(img)/255.0
        
        return preprocessed_img