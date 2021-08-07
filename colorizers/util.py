import numpy as np
import paddle
import paddle.nn.functional as F
from PIL import Image
from skimage import color


def load_img(img_path):
    out_np = np.asarray(Image.open(img_path))
    if out_np.ndim == 2:
        out_np = np.tile(out_np[:, :, None], 3)
    return out_np


def resize_img(img, HW=(256, 256), resample=3):
    return np.asarray(Image.fromarray(img).resize((HW[1], HW[0]), resample=resample))


def preprocess_img(img_rgb_orig, HW=(256, 256), resample=3):
    # return original size L and resized L as torch Tensors
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)

    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    img_l_orig = img_lab_orig[:, :, 0]
    img_l_rs = img_lab_rs[:, :, 0]

    tens_orig_l = paddle.to_tensor(img_l_orig, dtype=paddle.float32).unsqueeze((0, 1))
    tens_rs_l = paddle.to_tensor(img_l_rs, dtype=paddle.float32).unsqueeze((0, 1))

    return (tens_orig_l, tens_rs_l)


def postprocess_tens(tens_orig_l, out_ab, mode="bilinear"):
    # tens_orig_l 	1 x 1 x H_orig x W_orig
    # out_ab 		1 x 2 x H x W

    HW_orig = tens_orig_l.shape[2:]
    HW = out_ab.shape[2:]

    # call resize function if needed
    if HW_orig[0] != HW[0] or HW_orig[1] != HW[1]:
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode="bilinear")
    else:
        out_ab_orig = out_ab

    out_lab_orig = paddle.concat((tens_orig_l, out_ab_orig), axis=1)

    return color.lab2rgb(out_lab_orig.cpu().numpy()[0, ...].transpose((1, 2, 0)))
