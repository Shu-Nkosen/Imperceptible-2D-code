import sys
import cv2
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np


def compute_metrics(ref_path: str, target_path: str):
    ref = cv2.imread(ref_path)
    tgt = cv2.imread(target_path)
    if ref is None:
        raise FileNotFoundError(f"参照画像が読み込めません: {ref_path}")
    if tgt is None:
        raise FileNotFoundError(f"比較画像が読み込めません: {target_path}")
    if ref.shape != tgt.shape:
        raise ValueError(f"画像サイズが一致しません: {ref.shape} vs {tgt.shape}")

    psnr = cv2.PSNR(ref, tgt)

    gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    gray_tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY)
    ssim, _ = compare_ssim(gray_ref, gray_tgt, full=True)

    return psnr, ssim


def main():
    if len(sys.argv) != 3:
        print("Usage: python PSNR.py <reference_image> <target_image>")
        return

    psnr, ssim = compute_metrics(sys.argv[1], sys.argv[2])
    print(f"PSNR: {psnr:.4f} dB")
    print(f"SSIM: {ssim:.6f}")


if __name__ == "__main__":
    main()