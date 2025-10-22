import cv2
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np

def upscale_and_resize(image, target_width, target_height, scale=5):
    """
    解像度を上げてリサイズする関数。
    """
    # 解像度をスケーリング
    high_res_width = target_width * scale
    high_res_height = target_height * scale
    
    # リサイズ
    resized_image = cv2.resize(image, (high_res_width, high_res_height), interpolation=cv2.INTER_CUBIC)
    return resized_image

# オリジナル画像と比較画像を読み込む
# img_2 = cv2.imread('LocalNews10.png')


for i in range(5, 51, 5): 

    image_path = "Boat"
    img_org = cv2.imread(f'{image_path}resize.png')

    img_1 = cv2.imread(f'{image_path}{i}resize.png')


# PSNRの計算
    psnr_1 = cv2.PSNR(img_org, img_1)
# psnr_2 = cv2.PSNR(high_res_org, high_res_img_2)

# SSIMの計算（画像をグレースケールに変換）
    gray_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    gray_img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
# gray_img_2 = cv2.cvtColor(high_res_img_2, cv2.COLOR_BGR2GRAY)

    ssim_1, _ = compare_ssim(gray_org, gray_img_1, full=True)
# ssim_2, _ = compare_ssim(gray_org, gray_img_2, full=True)

# 結果を表示
    print(f"PSNR between high_res_org and high_res_img_1: {psnr_1}")
# print(f"PSNR between high_res_org and high_res_img_2: {psnr_2}")
    print(f"SSIM between high_res_org and high_res_img_1: {ssim_1}")
# print(f"SSIM between high_res_org and high_res_img_2: {ssim_2}")
