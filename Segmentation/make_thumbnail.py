import torch
from thumbnail import Thumbnail
from make_mask_img import make_mask_img
from make_thumbnail_fg import make_thumbnail_fg
from make_thumbnail_bg1 import make_thumbnail_bg1
from make_thumbnail_bg2 import make_thumbnail_bg2


a = Thumbnail(q1[:5].tolist(), step=5)
a.forward()
vlog_message = "QWER VLOG"

img_case, mask_img = make_mask_img(outputs=a.outputs, input_data=a.input_data)

if img_case != 4:
    dst = make_thumbnail_fg(img_case, mask_img)
    dst = make_thumbnail_bg1(dst1=dst, bg_image=a.background_img1, bg_c="sky", text_f="base", text_c="white",text=vlog_message, font_scale=2, font_thickness=2)
else:
    background_img_list = []

    for i in range(len(a.background_img)):
        background_img_list.append(a.tt(a.background_img[i]))

    torch_bg_list = [a.transform(tmp) for tmp in background_img_list]

    with torch.no_grad():
        background_output = a.model(torch_bg_list)
    dst = make_thumbnail_bg2(background_img=a.background_img, background_output=background_output, text_f="base", text_c="white", text=vlog_message, font_scale=2, font_thickness=2)