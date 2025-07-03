import cv2
import math
import numpy as np
import os, random
import os.path as osp
import torch
from pathlib import Path
import torch.utils.data as data
from basicsr.data import degradations as degradations
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from tqdm import tqdm

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def gen_one_lq(gt_path, filename, output_path):
    kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    blur_kernel_size = 21
    noise_range = [1, 30]
    blur_sigma = [0.2, 3.0]
    file_client = FileClient()

    gt_path = gt_path + '/' + filename
    img_bytes = file_client.get(gt_path)
    img_gt = imfrombytes(img_bytes, float32=True)   # [0, 1]
    #cv2.imwrite(output_path + '/gt/' + filename, img_gt * 255.0)

    h, w, _ = img_gt.shape

    # ------------------------ generate lq image ------------------------ #
    # blur
    kernel = degradations.random_mixed_kernels(
        kernel_list,
        kernel_prob,
        blur_kernel_size,
        blur_sigma,
        blur_sigma, [-math.pi, math.pi],
        noise_range=None)
    img_lq = cv2.filter2D(img_gt, -1, kernel)
    # downsample
    scale = 4
    img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
    # noise
    if noise_range is not None:
        img_lq = degradations.random_add_gaussian_noise(img_lq, noise_range)

    cv2.imwrite(output_path + filename, img_lq * 255.0)  # 将图像保存为 output_image.jpg
    print("Save " + filename)

def resize_and_save_images(src_folder, dest_folder):
    """
    遍历源文件夹下的所有子文件夹，将图片resize到指定大小，并保存到目标文件夹中
    :param src_folder: 原始文件夹路径
    :param dest_folder: 目标文件夹路径
    """
    # 遍历源文件夹的所有子文件夹
    for subdir, _, files in tqdm(os.walk(src_folder)):
        for file in files:
            # 只处理图片文件（根据后缀名判断）
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                # 获取源文件的完整路径
                src_file_path = os.path.join(subdir, file)
                
                # 计算目标文件夹中对应的子文件夹路径
                relative_path = os.path.relpath(subdir, src_folder)
                dest_subdir = os.path.join(dest_folder, relative_path)
                
                # 如果目标子文件夹不存在，创建它
                Path(dest_subdir).mkdir(parents=True, exist_ok=True)
                
                # 读取图像
                file_client = FileClient()
                img_bytes = file_client.get(src_file_path)
                img = imfrombytes(img_bytes, float32=True)

                # 如果读取失败，跳过该文件
                if img is None:
                    print(f"无法读取图像: {src_file_path}")
                    continue
                
                # 调整图像
                kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
                kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
                blur_kernel_size = 21
                noise_range = [1, 30]
                blur_sigma = [0.2, 3.0]

                h = 256
                w = 256

                # ------------------------ generate lq image ------------------------ #
                # blur
                kernel = degradations.random_mixed_kernels(
                    kernel_list,
                    kernel_prob,
                    blur_kernel_size,
                    blur_sigma,
                    blur_sigma, [-math.pi, math.pi],
                    noise_range=None)
                img_lq = cv2.filter2D(img, -1, kernel)
                # downsample
                scale = 4
                img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
                # noise
                if noise_range is not None:
                    img_lq = degradations.random_add_gaussian_noise(img_lq, noise_range)

                # 构造保存路径
                dest_file_path = os.path.join(dest_subdir, file)
                
                # 保存调整大小后的图像
                cv2.imwrite(dest_file_path, img_lq * 255.0)


def main(flag):
    if flag:
        folder_path = '/nfs5/xfy/test_images/gt'
        output_path = '/nfs5/xfy/test_images/lq'
        resize_and_save_images(folder_path, output_path)
    else:
        folder_path = '/home/xfy/Resshift/ResShift/testdata/Val_materials/gt'
        output_path = '/home/xfy/Resshift/ResShift/testdata/Val_materials/lq/'
        for filename in os.listdir(folder_path):
            # 检查是否是文件（排除子文件夹）
            if os.path.isfile(os.path.join(folder_path, filename)):
                gen_one_lq(folder_path, filename, output_path)

main(flag=True)