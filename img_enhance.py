import numpy as np
import cv2

import random
import math
import os
import threading
import shutil
from tqdm import tqdm


def format_path(file_path):
    current_path = os.path.abspath(file_path)
    return current_path.replace('\\', '/')


def find_files(folder):
    all_files=[]
    dirPath = format_path(folder)
    dirs = os.listdir(dirPath)  # 查找该层文件夹下所有的文件及文件夹，返回列表
    for currentFile in dirs:  # 遍历列表
        currentFile=format_path(os.path.join(dirPath,currentFile))
        if os.path.isfile(currentFile):
            all_files.append(currentFile)
        if os.path.isdir(currentFile):  # 如果是目录则递归，继续查找该目录下的文件
            temp=find_files(currentFile)
            for d in temp:
                all_files.append(d)
    return all_files


# 提取目录下所有类型文件
def find_files_type(folder,types):
    all_files = []
    dirPath = format_path(folder)
    if os.path.exists(dirPath) is False:
        print("{:s} not exist".format(dirPath))
        return
    dirs = os.listdir(dirPath)  # 查找该层文件夹下所有的文件及文件夹，返回列表
    for currentFile in dirs:  # 遍历列表
        currentFile = format_path(os.path.join(dirPath, currentFile))
        if os.path.isfile(currentFile):
            for type in types:
                if len(currentFile)>len(type) and currentFile[-len(type):] == type:
                    all_files.append(currentFile)
        if os.path.isdir(currentFile):  # 如果是目录则递归，继续查找该目录下的文件
            temp = find_files(currentFile)
            for d in temp:
                for type in types:
                    aa=d[-len(type):]
                    if len(d) > len(type) and d[-len(type):] == type:
                        all_files.append(d)
    return all_files

# 伽马变换
def gamma(image,row_path,name):
    fgamma = 2
    image_gamma = np.uint8(np.power((np.array(image) / 255.0), fgamma) * 255.0)
    cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
    cv2.convertScaleAbs(image_gamma, image_gamma)

    cv2.imwrite(os.path.join(row_path, "{}_{}.jpg".format(name[:-4], 'gamma')), image_gamma)
    # shutil.copyfile(os.path.join(row_path, "{}.txt".format(name[:-4])), os.path.join(row_path, "{}_{}.txt".format(name[:-4], 'gamma')))
    return image_gamma


# 限制对比度自适应直方图均衡化CLAHE
def clahe(image,row_path,name):
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    image_clahe = cv2.merge([b, g, r])
    cv2.imwrite(os.path.join(row_path, "{}_{}.jpg".format(name[:-4], 'clahe')), image_clahe)
    # shutil.copyfile(os.path.join(row_path, "{}.txt".format(name[:-4])),
    #                 os.path.join(row_path, "{}_{}.txt".format(name[:-4], 'clahe')))
    return image_clahe


def replaceZeroes(data,row_path,name):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


# retinex SSR
def SSR(src_img, size,row_path,name):
    L_blur = cv2.GaussianBlur(src_img, (size, size), 0)
    img = replaceZeroes(src_img)
    L_blur = replaceZeroes(L_blur)

    dst_Img = cv2.log(img / 200.0)
    dst_Lblur = cv2.log(L_blur / 200.0)
    dst_IxL = cv2.multiply(dst_Img, dst_Lblur)
    log_R = cv2.subtract(dst_Img, dst_IxL)

    dst_R = cv2.normalize(log_R, None, 0, 200, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8


def SSR_image(image,row_path,name):
    size = 3
    b_gray, g_gray, r_gray = cv2.split(image)
    b_gray = SSR(b_gray, size)
    g_gray = SSR(g_gray, size)
    r_gray = SSR(r_gray, size)
    result = cv2.merge([b_gray, g_gray, r_gray])
    return result


# retinex MMR
def MSR(img, scales,row_path,name):
    weight = 1 / 3.0
    scales_size = len(scales)
    h, w = img.shape[:2]
    log_R = np.zeros((h, w), dtype=np.float32)

    for i in range(scales_size):
        img = replaceZeroes(img)
        L_blur = cv2.GaussianBlur(img, (scales[i], scales[i]), 0)
        L_blur = replaceZeroes(L_blur)
        dst_Img = cv2.log(img / 255.0)
        dst_Lblur = cv2.log(L_blur / 255.0)
        dst_Ixl = cv2.multiply(dst_Img, dst_Lblur)
        log_R += weight * cv2.subtract(dst_Img, dst_Ixl)

    dst_R = cv2.normalize(log_R, None, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8


def MSR_image(image,row_path,name):
    scales = [15, 101, 301]  # [3,5,9]
    b_gray, g_gray, r_gray = cv2.split(image)
    b_gray = MSR(b_gray, scales)
    g_gray = MSR(g_gray, scales)
    r_gray = MSR(r_gray, scales)
    result = cv2.merge([b_gray, g_gray, r_gray])
    return result


def FOG(image,row_path,name):
    blur1 = cv2.blur(image, (5, 5))
    blur2 = cv2.blur(image, (5, 5), 0)
    median = cv2.medianBlur(image, 5)
    blur3 = cv2.bilateralFilter(image, 9, 75, 75)
    img = [blur1,blur2,median,blur3]
    ra = random.randint(0, 3)
    out = img[ra]
    cv2.imwrite(os.path.join(row_path, "{}_{}.jpg".format(name[:-4],'FOG')), out)
    # shutil.copyfile(os.path.join(row_path, "{}.txt".format(name[:-4])),
    #                 os.path.join(row_path, "{}_{}.txt".format(name[:-4], 'FOG')))

    return out


def rand_bright(img,row_path,name):
    dsts = []
    h, w, c = img.shape
    a = [0.65,0.7,0.75]
    g = [20,30,35]
    for i in a:
        for j in g:
            white = np.ones([h, w, c], img.dtype)*255
            dst = cv2.addWeighted(img, i, white, 1-i, j)
            dsts.append(dst)
    ra = random.randint(0, len(dsts)-1)
    bri = dsts[ra]
    cv2.imwrite(os.path.join(row_path, "{}_{}.jpg".format(name[:-4], 'rand_bright_img')), bri)
    # shutil.copyfile(os.path.join(row_path, "{}.txt".format(name[:-4])),
    #                 os.path.join(row_path, "{}_{}.txt".format(name[:-4], 'rand_bright_img')))
    return bri


def ran_noise(image,row_path,name):
    no_img = []
    # 设置添加椒盐噪声的数目比例
    s_vs_p = [0.3,0.4,0.5,0.6]
    amount = [0.001,0.002,0.003]
    noisy_img = np.copy(image)
    # 添加salt噪声
    for i in s_vs_p:
        for j in amount:
            num_salt = np.ceil(j * image.size * i)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            noisy_img[coords[0], coords[1], :] = [255, 255, 255]
            num_pepper = np.ceil(j * image.size * (1. - i))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            noisy_img[coords[0], coords[1], :] = [0, 0, 0]
            no_img.append(noisy_img)
    ra = random.randint(0, len(no_img) - 1)
    n_img = no_img[ra]
    return n_img


def post_process(img_path):
    all_img  = find_files_type(img_path, [".jpg", ".bmp"])
    for i_p in tqdm(all_img):
        row_path,name = os.path.split(i_p)
        img = cv2.imread(i_p)

        t1 = threading.Thread(target=gamma(img,row_path,name))
        t2 = threading.Thread(target=clahe(img,row_path,name))
        t3 = threading.Thread(target=FOG(img,row_path,name))
        t4 = threading.Thread(target=rand_bright(img,row_path,name))


img_path = r"F:\UAV_net\data\uav_data\img"
post_process(img_path)