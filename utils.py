import math
import numpy as np
import torch
import config
import cv2

feature_size_h = config.feature_size_h
feature_size_w = config.feature_size_w
stride_h = config.stride_h
stride_w = config.stride_w


def mask_to_onehot(mask):
    palette = [2, 0, 1]
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        if colour == 0:
            class_map = np.zeros((512, 512), dtype="uint8")
        else:
            class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    semantic_map = np.array(semantic_map, dtype="uint8")
    return semantic_map


def dis2conf(dis, dth):
    alpha = 2
    if dis <= dth:
        conf = (math.exp(alpha * (1 - dis / dth)) - 1) / (math.exp(alpha) - 1)
    else:
        conf = 0
    return conf


# Calculate the proximity between two stair lines
# Input format: (x1, y1, x2, y2)
def calculate_cross_line(line1, line2):
    d1 = (line1[0] - line2[0]) ** 2 + (line1[1] - line2[1]) ** 2
    d2 = (line1[0] - line2[2]) ** 2 + (line1[1] - line2[3]) ** 2
    D1 = (min(d1, d2)) ** 0.5
    d3 = (line1[2] - line2[0]) ** 2 + (line1[3] - line2[1]) ** 2
    d4 = (line1[2] - line2[2]) ** 2 + (line1[3] - line2[3]) ** 2
    D2 = (min(d3, d4)) ** 0.5
    return dis2conf(D1, 1) / 2 + dis2conf(D2, 1) / 2


def Draw_results(img, fb1, fb2, fr1, fr2, mask, conf=0.5, img_size=512):
    height, width = img.shape[0], img.shape[1]
    padh = (width - height) // 2
    for i in range(feature_size_h):
        for j in range(feature_size_w):
            # if contains object
            if fb1[i][j] >= conf:
                x0 = j * stride_w
                y0 = i * stride_h
                x1 = x0 + int(fb2[0][i][j] * stride_w)
                y1 = y0 + int(fb2[1][i][j] * stride_h)
                x2 = x0 + int(fb2[2][i][j] * stride_w)
                y2 = y0 + int(fb2[3][i][j] * stride_h)
                # transfer the coordinate in size (512, 512) to the coordinate in original size (height, width)
                xr1 = int(x1 / img_size * width)
                yr1 = int(y1 / img_size * width) - padh
                xr2 = int(x2 / img_size * width)
                yr2 = int(y2 / img_size * width) - padh
                xr1, yr1, xr2, yr2 = min(xr1, width), min(yr1, height), min(xr2, width), min(yr2, height)
                cv2.line(img, (xr1, yr1), (xr2, yr2), (255, 0, 0), 2)

            # if contains object
            if fr1[i][j] >= conf:
                x0 = j * stride_w
                y0 = i * stride_h
                x1 = x0 + int(fr2[0][i][j] * stride_w)
                y1 = y0 + int(fr2[1][i][j] * stride_h)
                x2 = x0 + int(fr2[2][i][j] * stride_w)
                y2 = y0 + int(fr2[3][i][j] * stride_h)

                xr1 = int(x1 / img_size * width)
                yr1 = int(y1 / img_size * width) - padh
                xr2 = int(x2 / img_size * width)
                yr2 = int(y2 / img_size * width) - padh
                xr1, yr1, xr2, yr2 = min(xr1, width), min(yr1, height), min(xr2, width), min(yr2, height)
                cv2.line(img, (xr1, yr1), (xr2, yr2), (0, 0, 255), 2)
    # Process predicted mask
    _, pred_mask = torch.max(mask, 1)
    pred_mask = pred_mask.cpu().data
    pred_mask = pred_mask.permute(1, 2, 0).numpy()
    pred_mask = mask_to_onehot(pred_mask) * 255
    # Resize and padded
    pred_mask = cv2.resize(pred_mask, (width, width))
    pred_mask = pred_mask[padh: padh + height, :]
    # Draw mask
    img = cv2.addWeighted(img, 0.7, pred_mask, 0.3, 0)
    return img
