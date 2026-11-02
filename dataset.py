import os
import glob
import random
import numpy as np
import cv2
from PIL import Image

# 大图防内存警告
Image.MAX_IMAGE_PIXELS = 1000000000

dataset_dir = '/mnt/data4/膀胱镜诊断/dataset/all-two/'
train_dir = '/mnt/data4/膀胱镜诊断/dataset/train/'
valid_dir = '/mnt/data4/膀胱镜诊断/dataset/valid/'
test_dir = '/mnt/data4/膀胱镜诊断/dataset/test/'

train_per = 0.7
valid_per = 0.15
test_per = 0.15

# ================= 黑边裁剪参数 =================
BLACK_THRESHOLD = 15      # 灰度低于此认为黑色，如果边缘组织被裁掉，可以调低到 10
PADDING = 10              # 裁剪后四周保留的像素
MAKE_SQUARE = True        # 是否强行裁剪成正方形（推荐True，后续Resize不会变形）


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


def crop_black_border(img, threshold=BLACK_THRESHOLD, padding=PADDING, make_square=MAKE_SQUARE):
    """
    裁剪内窥镜黑边，自动忽略角落的小图标/水印。
    """
    arr = np.array(img)
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr

    # 1. 二值化：非黑区域设为 255
    mask = (gray > threshold).astype(np.uint8) * 255

    # 2. 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return img, img.size  # 全黑

    # 3. 找面积最大的前景区域（排除了背景 0）
    max_area = 0
    best_idx = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            best_idx = i

    if best_idx == 0:
        return img, img.size

    # 4. 提取最大区域的 Bounding Box
    x = stats[best_idx, cv2.CC_STAT_LEFT]
    y = stats[best_idx, cv2.CC_STAT_TOP]
    w = stats[best_idx, cv2.CC_STAT_WIDTH]
    h = stats[best_idx, cv2.CC_STAT_HEIGHT]

    cx = x + w // 2
    cy = y + h // 2

    # 5. 裁剪区域计算
    if make_square:
        # 变成正方形，边长取最大边
        side = max(w, h)
        xmin = cx - side // 2
        ymin = cy - side // 2
        xmax = xmin + side
        ymax = ymin + side
    else:
        xmin, ymin, xmax, ymax = x, y, x + w, y + h

    # 加 padding
    xmin -= padding
    ymin -= padding
    xmax += padding
    ymax += padding

    # 限制边界
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(arr.shape[1], xmax)
    ymax = min(arr.shape[0], ymax)

    cropped_img = img.crop((xmin, ymin, xmax, ymax))
    return cropped_img, img.size


def save_image(img, out_path):
    ext = os.path.splitext(out_path)[1].lower()
    if ext in ('.jpg', '.jpeg'):
        img.convert('RGB').save(out_path, quality=95)
    else:
        img.save(out_path)


if __name__ == '__main__':
    IMG_EXTS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')

    for root, dirs, files in os.walk(dataset_dir):
        for sDir in dirs:
            imgs_list = []
            for ext in IMG_EXTS:
                imgs_list.extend(glob.glob(os.path.join(root, sDir, ext)))

            imgs_list = sorted(imgs_list)
            random.seed(666)
            random.shuffle(imgs_list)
            imgs_num = len(imgs_list)

            if imgs_num == 0:
                print(f'⚠️ 类别 {sDir} 没有图像，跳过')
                continue

            train_point = int(imgs_num * train_per)
            valid_point = int(imgs_num * (train_per + valid_per))

            n_train = n_valid = n_test = 0
            n_cropped = 0

            for i in range(imgs_num):
                if i < train_point:
                    out_dir = os.path.join(train_dir, sDir)
                    n_train += 1
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sDir)
                    n_valid += 1
                else:
                    out_dir = os.path.join(test_dir, sDir)
                    n_test += 1

                makedir(out_dir)
                out_path = os.path.join(out_dir, os.path.basename(imgs_list[i]))

                try:
                    img = Image.open(imgs_list[i]).convert('RGB')
                    cropped, orig_size = crop_black_border(img)
                    if cropped.size != orig_size:
                        n_cropped += 1
                    save_image(cropped, out_path)
                except Exception as e:
                    print(f'❌ 处理失败 {imgs_list[i]}: {e}')

            print(f'Class: {sDir}, 总数: {imgs_num}, train: {n_train}, valid: {n_valid}, test: {n_test}, 裁掉黑边: {n_cropped}')