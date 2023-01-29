import cv2
import torch
import torchvision
import numpy as np
import copy
import random
from google.colab import drive
from google.colab.patches import cv2_imshow
from torchvision.io.image import read_image
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights



def make_mask(outputs):
    img_num = len(outputs)
    tmp_dic = {}
    prior_label = [1, 18, 17]
    del_list = []
    real_dic = {}
    score_dic = {}
    proportion = 0.1
    proba_threshold = 0.5
    mask_img = []
    img_case = -1
    

    for i in prior_label:
        real_dic[i] = -1

    # boxes 버리고 각 이미지에서 레이블 중복되는 거 버림
    for i in range(img_num):
        # del outputs[i]["boxes"]   ##
        outputs[i]["labels"] = outputs[i]["labels"].detach().numpy()
        outputs[i]["scores"] = outputs[i]["scores"].detach().numpy()
        outputs[i]["masks"] = torch.squeeze(outputs[i]["masks"], 1)
        outputs[i]["masks"] = outputs[i]["masks"].detach().numpy()
        del_idx = []

        for j in range(1, len(outputs[i]["labels"])):
            if outputs[i]["labels"][j] in outputs[i]["labels"][:j]:
                del_idx.append(j)

        outputs[i]["labels"] = np.delete(outputs[i]["labels"], del_idx)
        outputs[i]["scores"] = np.delete(outputs[i]["scores"], del_idx)

        outputs[i]["masks"] = outputs[i]["masks"].reshape(outputs[i]["masks"].shape[0], -1)
        outputs[i]["masks"] = np.delete(outputs[i]["masks"], del_idx, axis=0)

    # tmp_dic에 레이블당 score가 가장 높은 애들을 저장
    for i in range(img_num):
        for j in range(len(outputs[i]["labels"])):
            if outputs[i]["labels"][j] in prior_label:
                if outputs[i]["labels"][j] not in tmp_dic:
                    tmp_dic[outputs[i]["labels"][j]] = [outputs[i]["scores"][j]]
                else:
                    tmp_dic[outputs[i]["labels"][j]].append(outputs[i]["scores"][j])

    for i in list(tmp_dic.keys()):
        del_list = []
        for j in tmp_dic[i]:
            if j < 0.7:
                del_list.append(j)

        for d in del_list:
            tmp_dic[i].remove(d)

        tmp_dic[i].sort(reverse=True)
    
    if len(tmp_dic[1]) != 0:   # 사람 O
        if len(tmp_dic[18]) == 0 and len(tmp_dic[17]) == 0:   # 사람 O, 개나 고양이 X
            img_case = 1
            while len(tmp_dic[1]) > 3:
                tmp_dic[1].pop()
        else:   # 사람 O, 개나 고양이 O
            img_case = 2
            if len(tmp_dic[18]) != 0:
                tmp_dic[18] = [max(tmp_dic[18])]
            if len(tmp_dic[17]) != 0:
                tmp_dic[17] = [max(tmp_dic[17])]
            while len(tmp_dic[1]) > 2:
                tmp_dic[1].pop()

    else:   # 사람 X
        if len(tmp_dic[18]) == 0 and len(tmp_dic[17]) == 0:   # 사람 X, 개나 고양이 X
            img_case = 4
        else:   # 사람 X, 개나 고양이 O
            img_case = 3
            while len(tmp_dic[18]) > 2:
                tmp_dic[18].pop()
            while len(tmp_dic[17]) > 2:
                tmp_dic[17].pop()
            while len(tmp_dic[18] + tmp_dic[17]) < 2:
                min_tmp = tmp_dic[18] + tmp_dic[17]
                if min_tmp in tmp_dic[17]:
                    tmp_dic[17].remove(min_tmp)
                else:
                    tmp_dic[18].remove(min_tmp)

    # 레이블당 score가 가장 높은 애들을 제외하고 나머지는 삭제
    for i in range(img_num):
        del_idx = []
        for j in range(len(outputs[i]["labels"])):
            if outputs[i]["labels"][j] in tmp_dic.keys():
                if outputs[i]["scores"][j] not in tmp_dic[outputs[i]["labels"][j]]:
                    del_idx.append(j)
            else:
                del_idx.append(j)
        outputs[i]["labels"] = np.delete(outputs[i]["labels"], del_idx)
        outputs[i]["scores"] = np.delete(outputs[i]["scores"], del_idx)
        outputs[i]["masks"] = np.delete(outputs[i]["masks"], del_idx, axis=0)
        outputs[i]["masks"] = outputs[i]["masks"].reshape(outputs[i]["masks"].shape[0], tmp_height, tmp_width)

    for i in range(img_num):
        for j in range(len(outputs[i]["labels"])):
            outputs[i]["masks"][j] = np.where(outputs[i]["masks"][j] < 0.25, 0, 255)
            dst = input_data[i][0].copy()
            dst = cv2.bitwise_and(dst, dst, mask=outputs[i]["masks"][j].astype("uint8"))
            mask_img.append(dst)

    return real_dic, img_case, mask_img


def make_thumbnail_fg(mask_img):
    img_list = []
    mask_img_copy = mask_img

    for i in range(len(mask_img)):
        tmp = cv2.cvtColor(mask_img[i], cv2.COLOR_BGR2GRAY)
        _, tmp = cv2.threshold(tmp, 0.9, 255, cv2.THRESH_BINARY)

        cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp)

        for j in range(1, cnt):
            (x, y, w, h, area) = stats[j]

            if area / tmp_img_size < 1 / 20:
                tmp[y:y+h, x:x+w] = 0
            
        a, b = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(a) != 0:
            cv2.drawContours(mask_img_copy[i], a, -1, (255, 255, 255), 11)
            tmp = cv2.bitwise_and(mask_img_copy[i], mask_img_copy[i], mask=tmp)
            img_list.append(tmp)
        

    for i in range(len(img_list)):
        tmp = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
        _, tmp = cv2.threshold(tmp, 0.9, 255, cv2.THRESH_BINARY)

        cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp)
        (x, y, w, h, area) = stats[1]

        m = np.float32([[1, 0, -x], [0, 1, -y]])

        img_list[i] = cv2.warpAffine(img_list[i], m, (0, 0))

        if i == 0:
            if area / tmp_img_size > 1/ 2:
                img_list[i] = cv2.resize(img_list[i], (None, None), fx=0.65, fy=0.65, interpolation=cv2.INTER_AREA)
            elif area / tmp_img_size > 1 / 3:
                img_list[i] = cv2.resize(img_list[i], (None, None), fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
            elif area / tmp_img_size > 1 / 6:
                img_list[i] = cv2.resize(img_list[i], (None, None), fx=0.85, fy=0.85, interpolation=cv2.INTER_AREA)
            elif area / tmp_img_size > 1 / 10:
                img_list[i] = cv2.resize(img_list[i], (None, None), fx=0.95, fy=0.95, interpolation=cv2.INTER_AREA)
            else:
                img_list[i] = cv2.resize(img_list[i], (None, None), fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)
        elif i == 1:
            if area / tmp_img_size > 1/ 2:
                img_list[i] = cv2.resize(img_list[i], (None, None), fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
            elif area / tmp_img_size > 1 / 3:
                img_list[i] = cv2.resize(img_list[i], (None, None), fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
            elif area / tmp_img_size > 1 / 6:
                img_list[i] = cv2.resize(img_list[i], (None, None), fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
            elif area / tmp_img_size > 1 / 10:
                img_list[i] = cv2.resize(img_list[i], (None, None), fx=0.9, fy=0.9, interpolation=cv2.INTER_AREA)
            else:
                img_list[i] = cv2.resize(img_list[i], (None, None), fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)

        elif i == 2 or i == 3:
            if area / tmp_img_size > 1/ 2:
                img_list[i] = cv2.resize(img_list[i], (None, None), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            elif area / tmp_img_size > 1 / 3:
                img_list[i] = cv2.resize(img_list[i], (None, None), fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA)
            elif area / tmp_img_size > 1 / 6:
                img_list[i] = cv2.resize(img_list[i], (None, None), fx=0.45, fy=0.45, interpolation=cv2.INTER_AREA)
            elif area / tmp_img_size > 1/ 10:
                img_list[i] = cv2.resize(img_list[i], (None, None), fx=0.85, fy=0.85, interpolation=cv2.INTER_AREA)
            elif area / tmp_img_size < 1 / 25:
                img_list[i] = cv2.resize(img_list[i], (None, None), fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)
        else:
            if area / tmp_img_size >= 1 / 2:
                img_list[i] = cv2.resize(img_list[i], (None, None), fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
            elif area / tmp_img_size > 1 / 3:
                img_list[i] = cv2.resize(img_list[i], (None, None), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
            elif area / tmp_img_size > 1 / 6:
                img_list[i] = cv2.resize(img_list[i], (None, None), fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
            elif area / tmp_img_size > 1 / 10:
                img_list[i] = cv2.resize(img_list[i], (None, None), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            elif area / tmp_img_size > 1 / 15:
                img_list[i] = cv2.resize(img_list[i], (None, None), fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
            elif area / tmp_img_size > 1 / 20:
                img_list[i] = cv2.resize(img_list[i], (None, None), fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
            elif area / tmp_img_size < 1 / 30:
                img_list[i] = cv2.resize(img_list[i], (None, None), fx=1.1, fy=1.1, interpolation=cv2.INTER_LINEAR)
    
    for i in range(len(img_list)):
        bg = np.zeros((tmp_height, tmp_width, 3), dtype=np.uint8)

        if img_list[i].shape[0] > bg.shape[0]:
            bg[:bg.shape[0], :bg.shape[1]] = img_list[i][:bg.shape[0], :bg.shape[1]]
        else:
            bg[:img_list[i].shape[0], :img_list[i].shape[1]] = img_list[i]

        img_list[i] = bg

    pos_dic = {}

    for i in range(len(img_list)):
        tmp = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
        _, tmp = cv2.threshold(tmp, 1, 255, cv2.THRESH_BINARY)

        cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp)
        centroids = centroids.astype(np.int64)
        (x, y, w, h, area) = stats[1]
        pos_dic[i] = [x, y, w, h, area, (centroids[1][0], centroids[1][1])]

        if i == 0:
            shift_x = 10
            shift_y = tmp_height - h

        elif i == 1:
            shift_x = tmp_width - w - 10
            shift_y = tmp_height - h
        
        elif i == 2:
            shift_x = pos_dic[0][2] + 10
            shift_y = tmp_height - pos_dic[0][3] - int(h / 3)

        elif i == 3:
            shift_x = tmp_width - w - 10
            shift_y = tmp_height - pos_dic[1][3] - h - 10

        else:
            if i == 4:
                shift_x = 20
                shift_y = 20
            else:
                # shift_x = pos_dic[0][2] + pos_dic[2][2] + 20
                # shift_x = (pos_dic[2][2] + pos_dic[4][2]) * int(1 + i / 10)
                # shift_y = pos_dic[i - 1][3] - int(pos_dic[i][3] / 2)
                shift_x = int(tmp_width / (15 - i) * i)
                shift_y += int(pos_dic[i][3] / 3)


        m = np.float32([[1, 0, int(shift_x)], [0, 1, int(shift_y)]])
        img_list[i] = cv2.warpAffine(img_list[i], m, (0, 0))


    dst1 = np.zeros((tmp_height, tmp_width, 3), np.uint8)

    for i in img_list[::-1]:
        # dst2 = cv2.add(dst2, i)
        # dst2 = cv2.bitwise_or(dst2, i)

        gray_tmp = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_tmp, 0.1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        masked_fg = cv2.bitwise_and(i, i, mask=mask)
        masked_bg = cv2.bitwise_and(dst1, dst1, mask=mask_inv)

        dst1 = masked_fg + masked_bg

    return dst1


def make_thumbnail_bg(dst1, bg_image=True, bg_c="sky", text_f="TRIPLEX", text_c="black",text="VLOG", font_scale=2, font_thickness=2):
    bg_color = {"sky": cv2.merge([np.full((tmp_height, tmp_width), 255, np.uint8), np.full((tmp_height, tmp_width), 204, np.uint8), np.full((tmp_height, tmp_width), 153, np.uint8)]),
            "pink": cv2.merge([np.full((tmp_height, tmp_width), 255, np.uint8), np.full((tmp_height, tmp_width), 51, np.uint8), np.full((tmp_height, tmp_width), 255, np.uint8)]),
            "red": cv2.merge([np.full((tmp_height, tmp_width), 0, np.uint8), np.full((tmp_height, tmp_width), 0, np.uint8), np.full((tmp_height, tmp_width), 255, np.uint8)]),
            }
    text_color = {"black": (0, 0, 0), "red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 255, 0), "white": (255, 255, 255)}
    text_font = {"SIMPLEX": cv2.FONT_HERSHEY_SIMPLEX, "TRIPLEX": cv2.FONT_HERSHEY_TRIPLEX, "ITALIC": cv2.FONT_ITALIC}

    if bg_image:
        dst2 = background_img1
    else:
        dst2 = bg_color[bg_c]

    text_size = cv2.getTextSize(text, text_font[text_f], font_scale, font_thickness)[0]
    text_x = int((dst2.shape[1] - text_size[0]) / 2)
    text_y = int((dst2.shape[0] - text_size[1]) / 4 * 3)
    dst1_gray = cv2.cvtColor(dst1, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(dst1_gray, 0.1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    masked_fg = cv2.bitwise_and(dst1, dst1, mask=mask)
    masked_bg = cv2.bitwise_and(dst2, dst2, mask=mask_inv)
    dst = masked_fg + masked_bg
    # dst = dst2 + masked_bg

    cv2.putText(dst, text, (text_x, text_y), text_font[text_f], font_scale, text_color[text_c], font_thickness)

    return dst


drive.mount("/content/gdrive")

q = np.load("/content/gdrive/MyDrive/Colab Notebooks/SKT_FLY_AI/project/image/bts_thumb_imgs.npy", allow_pickle=True)
input_data = q[:10].tolist()

original_height, original_width = input_data[0][0].shape[:2]
original_img_size = original_height * original_width

for i in input_data:
    if i[0].shape[1] > 1500:
        i[0] = cv2.resize(i[0], (None, None), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    elif i[0].shape[1] > 1000:
        i[0] = cv2.resize(i[0], (None, None), fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)


background_img3 = input_data[0]   # 세그먼트할 대상이 없을 경우 사용할 배경(3개 이미지중 첫번째)
background_img4 = input_data[len(input_data) // 2]   # 세그먼트할 대상이 없을 경우 사용할 배경(3개 이미지중 두번째)
background_img5 = input_data[-1]   # 세그먼트할 대상이 없을 경우 사용할 배경(3개 이미지중 세번째)

input_data.sort(reverse=True, key=lambda x:x[2])   # idx 1은 cps_score, idx 2는 frame_score, frame_score로 정렬

background_img1 = input_data.pop()[0]   # 세그먼트할 대상이 있는 경우 사용할 배경
background_img2 = input_data.pop(0)[0]   # 세그먼트할 대상이 없을 경우 사용할 배경(1개 이미지)
tmp_height, tmp_width = input_data[0][0].shape[:2]
tmp_img_size = tmp_height * tmp_width


weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
transform = weights.transforms()

model = maskrcnn_resnet50_fpn(weights=weights)
model = model.eval()


input_img_list = []
tt = torchvision.transforms.ToTensor()

for i in range(len(input_data)):
    input_img_list.append(tt(input_data[i][0]))


torch_img_list = [transform(tmp) for tmp in input_img_list]

outputs = model(torch_img_list)


real_dic, img_case, mask_img = make_mask(outputs)
dst1 = make_thumbnail_fg(mask_img)
dst2 = make_thumbnail_bg(dst1, bg_image=True, bg_c="sky", text_f="TRIPLEX", text_c="white", text="QWE's VLOG", font_scale=2, font_thickness=2)

cv2.imwrite("/content/gdrive/MyDrive/Colab Notebooks/SKT_FLY_AI/project/image/dst.jpg", dst2)
cv2.imwrite("./dst.jpg", dst2)



