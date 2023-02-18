import cv2
import numpy as np


def plus_sticker(category, dst):
    dst_copy = dst.copy()
    # a = cv2.imread("images/animal.png", cv2.IMREAD_COLOR)
    # b = cv2.imread("images/couple.png", cv2.IMREAD_COLOR)
    # c = cv2.imread("images/family.png", cv2.IMREAD_COLOR)
    # d = cv2.imread("images/food.png", cv2.IMREAD_COLOR)
    # e = cv2.imread("images/friends.png", cv2.IMREAD_COLOR)
    # f = cv2.imread("images/leisure.png", cv2.IMREAD_COLOR)
    # g = cv2.imread("images/scene.png", cv2.IMREAD_COLOR)


    if category == "animal":
        sticker = cv2.imread("images/animal.png", cv2.IMREAD_COLOR)
    elif category == "couple":
        sticker = cv2.imread("images/couple.png", cv2.IMREAD_COLOR)
    elif category == "family":
        sticker = cv2.imread("images/family.png", cv2.IMREAD_COLOR)
    elif category == "food":
        sticker = cv2.imread("images/food.png", cv2.IMREAD_COLOR)
    elif category == "friends":
        sticker = cv2.imread("images/friends.png", cv2.IMREAD_COLOR)
    elif category == "leisure":
        sticker = cv2.imread("images/leisure.png", cv2.IMREAD_COLOR)
    elif category == "scene":
        sticker = cv2.imread("images/scene.png", cv2.IMREAD_COLOR)

    bg = np.zeros(dst_copy.shape, np.uint8)
    bg[40:40+sticker.shape[0], 20:20+sticker.shape[1]] = sticker
    # m = np.float32([[1, 0, 3 * sticker.shape[1], [0, 1, sticker.shape[0]]]])
    # bg = cv2.warpAffine(bg, m, (0, 0))
    bg[20:20+sticker.shape[0], bg.shape[0] // 3 * 2 + sticker.shape[1] * 2:bg.shape[0] // 3 * 2 + sticker.shape[1] * 3] = sticker
    # warpAffine 고려할 것 
    # cv2_imshow(bg)
    # print()
    # cv2_imshow(dst_copy)
    # print()

    a = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    b = cv2.threshold(a, 1, 255, cv2.THRESH_BINARY)[1]
    dst_copy = cv2.bitwise_and(dst_copy, dst_copy, mask=cv2.bitwise_not(b))
    dst_copy = dst_copy + bg

    return dst_copy






