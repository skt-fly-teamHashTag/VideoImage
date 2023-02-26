import cv2
import numpy as np
from font_bg import poo_make_image


def make_thumbnail_daily(img, message):
    tmp_height, tmp_width = img.shape[:2]
    message = message
    bg = cv2.merge([np.full((tmp_height + 20, tmp_width + 20), 0, np.uint8), np.full((tmp_height + 20, tmp_width + 20), 212, np.uint8), np.full((tmp_height + 20, tmp_width + 20), 255, np.uint8)])
    bg[10:tmp_height + 10, 10:tmp_width + 10] = 0
    bg[10:tmp_height + 10, 10:tmp_width + 10] = img

    bg = poo_make_image(message, img)
    
    return bg