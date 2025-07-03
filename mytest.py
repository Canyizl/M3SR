import numpy as np
import cv2
import lpips
import torch
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import os
from tqdm import tqdm
import shutil
from utils import util_image

from datapipe.datasets import PairedData

# 1. PSNR 计算
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100  # 完全相同的图像，PSNR为无限大
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

# 2. SSIM 计算
def compute_ssim(img1, img2):
    ssim_value, _ = ssim(img1, img2, full=True)
    return ssim_value

# 3. LPIPS 计算
def compute_lpips(img1_path, img2_path):
    # 加载预训练的 LPIPS 网络
    lpips_net = lpips.LPIPS(net='alex')  # 使用 AlexNet 作为特征提取网络
    
    # 读取并转换图像
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    
    # 将图像转换为 tensor
    img1 = np.array(img1).astype(np.float32) / 255.0
    img2 = np.array(img2).astype(np.float32) / 255.0
    img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0)
    img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0)

    # 计算 LPIPS 值
    lpips_value = lpips_net(img1, img2)
    return lpips_value.item()

# 主函数
def evaluate_image_quality(original_img_path, reconstructed_img_path):
    # 读取原图和重建图像
    #print(original_img_path)
    #print(reconstructed_img_path)
    img1 = cv2.imread(original_img_path)
    img2 = cv2.imread(reconstructed_img_path)

    # PSNR 计算
    psnr_value = util_image.calculate_psnr(img1,img2, ycbcr=True)
    psnr_value_rgb = psnr(img1, img2)

    # SSIM 计算 (转换为灰度图计算)
    ssim_value = util_image.calculate_ssim(img1,img2)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim_value_gray = compute_ssim(img1_gray, img2_gray)

    # LPIPS 计算
    #lpips_value = compute_lpips(original_img_path, reconstructed_img_path)

    return psnr_value, ssim_value, psnr_value_rgb, ssim_value_gray#, lpips_value



def organize_images_by_prefix():
    """
    将源文件夹中的所有图片按照文件名_前的部分进行分类，转移到目标文件夹下的对应子文件夹中
    :param src_folder: 源文件夹路径，包含子图文件
    :param dest_folder: 目标文件夹路径，将会创建子文件夹并将图片分类存储
    """
    src_folder = 'samples_image/pred'  # 替换为源文件夹路径
    dest_folder = 'pred_images'  # 替换为目标文件夹路径
    # 确保目标文件夹存在
    os.makedirs(dest_folder, exist_ok=True)

    # 遍历源文件夹中的所有文件
    for file_name in tqdm(os.listdir(src_folder)):
        file_path = os.path.join(src_folder, file_name)
        
        # 只处理文件，跳过子文件夹
        if os.path.isfile(file_path):
            # 获取文件名的前缀（即 _ 前的部分）
            prefix = file_name.split('_')[0]
            
            # 创建以文件名前缀命名的子文件夹
            prefix_folder = os.path.join(dest_folder, prefix)
            os.makedirs(prefix_folder, exist_ok=True)
            
            # 构造目标文件路径
            dest_file_path = os.path.join(prefix_folder, file_name)
            
            # 将文件移动到对应的子文件夹中
            shutil.move(file_path, dest_file_path)

def calc_metrics():
    # 示例：使用图片路径进行评估
    original_img_folder = '/nfs5/xfy/test_images/gt' #'testdata/Val_materials/gt'  # 替换为原始图像路径
    reconstructed_img_folder = 'pred_images' #'testdata/Val_materials/pred'  # 替换为重建图像路径

    index = 0
    psnr_y_all = 0
    psnr_rgb_all = 0
    ssim_all = 0 
    ssim_gray_all = 0 
    lpips_all = 0
    for subdir, _, files in os.walk(original_img_folder):
        for file in tqdm(files):
            # 只处理图片文件（根据后缀名判断）
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                original_img_folder = '/nfs5/xfy/test_images/gt'
                reconstructed_img_folder = 'pred_images'
                prefix = file.split('_')[0]
                original_img_folder = os.path.join(original_img_folder, prefix)
                reconstructed_img_folder = os.path.join(reconstructed_img_folder, prefix)
                gt_image_path = os.path.join(original_img_folder, file)
                pred_image_path = os.path.join(reconstructed_img_folder, file)       
                psnr_y_t, ssim_t, psnr_rgb_t, ssim_gray_t = evaluate_image_quality(gt_image_path, pred_image_path)
                index += 1
                psnr_y_all += psnr_y_t
                psnr_rgb_all += psnr_rgb_t
                ssim_all += ssim_t
                ssim_gray_all += ssim_gray_t

    print("All_Number: ", index)
    print("Ychannel: \n")
    print("PSNR: ", psnr_y_all / index)
    print("SSIM: ", ssim_all / index)
    print("RGB: \n")
    print("PSNR: ", psnr_rgb_all / index)
    print("SSIM: ", ssim_gray_all / index)
    #print("lpips: ", lpips_all / index)

def main():
    #organize_images_by_prefix()
    #calc_metrics()

    dir_path = "/nfs5/xfy/magnifydataset/test/lowscale"
    dir_path_extra = "/nfs5/xfy/magnifydataset/test/highscale"
    dataloader = PairedData(dir_path=dir_path, dir_path_extra=dir_path_extra, im_exts="tif", length = 32)

    print(dataloader.length)

    for i,data in enumerate(dataloader):
        lq = np.array(data['lq']).transpose(1,2,0) * 255.0
        gt = np.array(data['gt']).transpose(1,2,0) * 255.0
        scale = np.array(data['scale'])
        print(lq.shape, gt.shape, scale[0], scale[1])
        cv2.imwrite("testdata/lq_" + str(i) + ".jpg",lq)
        cv2.imwrite("testdata/gt_" + str(i) + ".jpg",gt)
        if i > 5:
            exit()
        

main()