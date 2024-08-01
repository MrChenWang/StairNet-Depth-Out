import os
import cv2
import numpy as np

import config

feature_size_h = config.feature_size_h
feature_size_w = config.feature_size_w


def convert_bbox2labels(line, path, feature_size_h, feature_size_w, img_size=512, flip=0):
    stride_h = img_size // feature_size_h
    stride_w = img_size // feature_size_w
    img = cv2.imread(path)
    # Each cell is denoted as (x1, y1, x2, y2, number of points, probability of Gaussian, cls) with 7 elements
    # We store blue and red lines with 2 arrays
    blue_labels = np.zeros((feature_size_h, feature_size_w, 7))
    red_labels = np.zeros((feature_size_h, feature_size_w, 7))
    for i in range(len(line) // 5):
        blue_points = []
        red_points = []
        # get the original label
        cls, x1, y1, x2, y2 = line[i * 5], line[i * 5 + 1], line[i * 5 + 2], line[i * 5 + 3], line[i * 5 + 4]
        # Prevent denominator from being 0
        if y1 == y2:
            y1 = y1 + 1
            y2 = y2 - 1
        # cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
        # for the 2 endpoints, the probability is 1
        if cls == 0:
            blue_points.append((x1 // stride_w, y1 // stride_h, x1, y1))
            blue_points.append((x2 // stride_w, y2 // stride_h, x2, y2))
        if cls == 1:
            red_points.append((x1 // stride_w, y1 // stride_h, x1, y1))
            red_points.append((x2 // stride_w, y2 // stride_h, x2, y2))
        sigma = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) // 2
        # slope of line
        k = (y1 - y2) / (x1 - x2)
        # get the intersection point between stair line and each vertical line
        for x in range(feature_size_w):
            if x1 < stride_w * x < x2:
                # fx = int(k*(stride*x-x1)+y1)
                fx = k * (stride_w * x - x1) + y1
                if cls == 0:
                    blue_points.append((x, int(fx // stride_w), stride_w * x, fx))
                if cls == 1:
                    red_points.append((x, int(fx // stride_w), stride_w * x, fx))
        # get the intersection point between stair line and each horizontal line
        for y in range(feature_size_h):
            if (y1 < stride_h * y < y2 and y1 < y2) or (y2 < stride_h * y < y1 and y1 > y2):
                fy = (stride_h * y - y1) / k + x1
                if fy % stride_w != 0:
                    if cls == 0:
                        blue_points.append((int(fy // stride_h), y, fy, stride_h * y))
                    if cls == 1:
                        red_points.append((int(fy // stride_h), y, fy, stride_h * y))

        # for each cell, find the contained points
        if cls == 0:
            for m in range(feature_size_h):
                for n in range(feature_size_w):
                    for k in range(len(blue_points)):
                        x = blue_points[k][2]
                        y = blue_points[k][3]
                        if n * stride_w <= x <= n * stride_w + stride_w and m * stride_h <= y <= m * stride_h + stride_h:
                            # transfer the real coordinate (x,y) to normalized coordinates
                            blue_labels[m, n, 2 * int(blue_labels[m, n, 4])] = x / stride_w - n
                            blue_labels[m, n, 2 * int(blue_labels[m, n, 4]) + 1] = y / stride_h - m
                            blue_labels[m, n, 4] += 1  # store the number of points
                            distance = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
                            if distance <= sigma:
                                blue_labels[m, n, 5] = np.exp(-0.5 * ((x - x1) ** 2 + (y - y1) ** 2) / sigma ** 2)
                            else:
                                blue_labels[m, n, 5] = np.exp(-0.5 * ((x - x2) ** 2 + (y - y2) ** 2) / sigma ** 2)
                            blue_labels[m, n, 6] = 0
        if cls == 1:
            for m in range(feature_size_h):
                for n in range(feature_size_w):
                    for k in range(len(red_points)):
                        x = red_points[k][2]
                        y = red_points[k][3]
                        if n * stride_w <= x <= n * stride_w + stride_w and m * stride_h <= y <= m * stride_h + stride_h:
                            # transfer the real coordinate (x,y) to normalized coordinates
                            red_labels[m, n, 2 * int(red_labels[m, n, 4])] = x / stride_w - n
                            red_labels[m, n, 2 * int(red_labels[m, n, 4]) + 1] = y / stride_h - m
                            red_labels[m, n, 4] += 1
                            distance = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
                            if distance <= sigma:
                                red_labels[m, n, 5] = np.exp(-0.5 * ((x - x1) ** 2 + (y - y1) ** 2) / sigma ** 2)
                            else:
                                red_labels[m, n, 5] = np.exp(-0.5 * ((x - x2) ** 2 + (y - y2) ** 2) / sigma ** 2)
                            red_labels[m, n, 6] = 1

    # drawing for validation
    heatmaps = np.zeros((feature_size_h, feature_size_w, 3), dtype="uint8")
    for m in range(feature_size_h):
        for n in range(feature_size_w):
            if blue_labels[m, n, 4] == 2:
                cv2.rectangle(img, (n * stride_w, m * stride_h), (n * stride_w + stride_w, m * stride_h + stride_h),
                              (255, 0, 255), 1)
                x1 = blue_labels[m, n, 0]
                y1 = blue_labels[m, n, 1]
                x2 = blue_labels[m, n, 2]
                y2 = blue_labels[m, n, 3]
                gaussian_value = blue_labels[m, n, 5]
                heatmaps[m, n, 0] = gaussian_value * 255
                heatmaps[m, n, 1] = gaussian_value * 255
                heatmaps[m, n, 2] = gaussian_value * 255
                cv2.line(img, (int((x1 + n) * stride_w), int((y1 + m) * stride_h)),
                         (int((x2 + n) * stride_w), int((y2 + m) * stride_h)), (255, 0, 0), 1)
            if red_labels[m, n, 4] == 2:
                cv2.rectangle(img, (n * stride_w, m * stride_h), (n * stride_w + stride_w, m * stride_h + stride_h),
                              (255, 0, 255), 1)
                x1 = red_labels[m, n, 0]
                y1 = red_labels[m, n, 1]
                x2 = red_labels[m, n, 2]
                y2 = red_labels[m, n, 3]
                gaussian_value = red_labels[m, n, 5]
                heatmaps[m, n, 0] = gaussian_value * 255
                heatmaps[m, n, 1] = gaussian_value * 255
                heatmaps[m, n, 2] = gaussian_value * 255
                cv2.line(img, (int((x1 + n) * stride_w), int((y1 + m) * stride_h)),
                         (int((x2 + n) * stride_w), int((y2 + m) * stride_h)), (0, 0, 255), 1)
    GrayImage = cv2.cvtColor(heatmaps, cv2.COLOR_BGR2GRAY)
    if flip == 0:
        cv2.imwrite(path.replace("images", "heatmaps").replace(".jpg", ".png"), GrayImage)
        cv2.imwrite(path.replace("images", "checkes"), img)
    return blue_labels, red_labels


def file_txt2photo(img_root, label_root):
    L_img = []
    L_label = []
    # file_handle = open('./data/train.txt', mode='w')
    for root, dirs, files in os.walk(label_root):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':  # 想要保存的文件格式
                L_img.append(os.path.join(img_root, os.path.splitext(file)[0] + '.jpg'))
                L_label.append(os.path.join(root, file))
                # file_handle.write(os.path.splitext(file)[0]+'\n')
    # file_handle.close()
    return L_img, L_label


def get_final_label(img_path, label_path, img_size=512, flip=0):
    L_img, L_label = file_txt2photo(img_path, label_path)
    for idx in range(len(L_label)):
        print(L_img[idx])
        # file_handle = open(L_label[idx].replace("labels", "final_labels32asymmetric"), mode='w')
        if flip == 1:
            file_handle = open(L_label[idx].replace("labels", "final_labels_flip32asymmetric"), mode='w')
        else:
            file_handle = open(L_label[idx].replace("labels", "final_labels32asymmetric"), mode='w')
        with open(L_label[idx]) as f:
            bbox = f.read().split('\n')
        bbox = [x.split() for x in bbox]
        bbox = [float(x) for y in bbox for x in y]
        if flip == 1:
            for i in range(len(bbox) // 5):
                bbox[i * 5 + 1] = img_size - bbox[i * 5 + 1]
                bbox[i * 5 + 3] = img_size - bbox[i * 5 + 3]
                if bbox[i * 5 + 3] < bbox[i * 5 + 1]:
                    bbox[i * 5 + 1], bbox[i * 5 + 3] = bbox[i * 5 + 3], bbox[i * 5 + 1]
                    bbox[i * 5 + 2], bbox[i * 5 + 4] = bbox[i * 5 + 4], bbox[i * 5 + 2]
        blue_labels, red_labels = convert_bbox2labels(bbox, L_img[idx], feature_size_h, feature_size_w, flip=flip)
        # write the final labels
        for i in range(feature_size_h):
            for j in range(feature_size_w):
                if blue_labels[i, j, 4] == 2:
                    info = str(int(blue_labels[i, j, 6])) + ' ' + str(
                        format(blue_labels[i, j, 0], '.3f')) + ' ' + str(
                        format(blue_labels[i, j, 1], '.3f')) + ' ' + str(
                        format(blue_labels[i, j, 2], '.3f')) + ' ' + str(
                        format(blue_labels[i, j, 3], '.3f')) + ' ' + str(
                        format(blue_labels[i, j, 5], '.3f')) + ' ' + str(i) + ' ' + str(j) + '\n'
                    file_handle.write(info)
                if red_labels[i, j, 4] == 2:
                    info = str(int(red_labels[i, j, 6])) + ' ' + str(
                        format(red_labels[i, j, 0], '.3f')) + ' ' + str(
                        format(red_labels[i, j, 1], '.3f')) + ' ' + str(
                        format(red_labels[i, j, 2], '.3f')) + ' ' + str(
                        format(red_labels[i, j, 3], '.3f')) + ' ' + str(
                        format(red_labels[i, j, 5], '.3f')) + ' ' + str(i) + ' ' + str(j) + '\n'
                    file_handle.write(info)
        file_handle.close()


if __name__ == "__main__":
    img_path, label_path = "data\\val\\images", "data\\val\\labels"
    get_final_label(img_path, label_path, flip=0)
