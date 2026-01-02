import cv2
import numpy as np
import os

# ======= パラメータ設定 =======
XX = 8
# 変化させるRGB量
BRIGHTNESS_DECREASE_POS = XX  # 黒部分の輝度減少量（normal）
BRIGHTNESS_DECREASE_NEG = -XX # 黒部分の輝度増加量（inv）
PRE_CLIP_MARGIN = 4           # 事前クリップマージン
# ==============================

def adjust_brightness_by_qr(image_path, qr_path, bright_decrease, clip_margin, invert=False, output_suffix=""):
    image = cv2.imread(image_path)
    qr_code = cv2.imread(qr_path, cv2.IMREAD_GRAYSCALE)

    if image is None or qr_code is None:
        print(f"Error: 画像の読み込みに失敗しました - {image_path} or {qr_path}")
        return

    # 入力画像をフルHD(1920x1080)に揃えて処理
    image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_AREA)
    
    img_h, img_w = image.shape[:2]
    square_size = img_h
    x_offset = (img_w - square_size) // 2
    
    qr_resized = cv2.resize(qr_code, (square_size, square_size))
    _, qr_binary = cv2.threshold(qr_resized, 127, 255, cv2.THRESH_BINARY)
    
    if invert:
        qr_binary = 255 - qr_binary

    black_mask = qr_binary == 0

    channel_info = [("R", 2), ("G", 1), ("B", 0)]
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    invert_str = "inv" if invert else "normal"

    for channel_name, channel_idx in channel_info:
        channel_image = image.astype(np.float32)

        # =========================================================================
        # 【修正】実験条件統一のため、いじらない色も含めて「全画素・全チャンネル」をクリップ
        # これで背景もROIも、RもGもBも、すべて [4, 251] スタートで統一されます。
        # =========================================================================
        if clip_margin > 0:
            channel_image = np.clip(channel_image, clip_margin, 255.0 - clip_margin)

        # ここから「1チャンネルだけ」を取り出して操作
        channel_region = channel_image[:, x_offset:x_offset + square_size, channel_idx]

        # 輝度変化の適用（指定チャンネルのみ）
        channel_region[black_mask] -= bright_decrease
        
        # 演算後の値を0-255に再クリップ
        channel_region = np.clip(channel_region, 0, 255)
        
        # 処理結果を書き戻す
        channel_image[:, x_offset:x_offset + square_size, channel_idx] = channel_region

        result = channel_image.astype(np.uint8)
        tag = output_suffix or f"{abs(int(bright_decrease))}_{invert_str}"
        output_name = f"{base_name}_{tag}{channel_name}.png"
        output_path = os.path.join(os.path.dirname(image_path), output_name)

        cv2.imwrite(output_path, result)
        print(f"保存完了: {output_name} (Pre-clip: {clip_margin})")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    qr_path = os.path.join(current_dir, "HP_QR.png")
    
    target_images = [
        "rice.png",
        "kosen.png",
        "nagaoka_fireworks.png",
        "hocho.png",
        "ex.png"
    ]
    
    modes = [
        {
            "bright_decrease": BRIGHTNESS_DECREASE_POS,
            "invert": False,
            "suffix": "normal",
        },
        {
            "bright_decrease": BRIGHTNESS_DECREASE_NEG,
            "invert": False,
            "suffix": "inv",
        },
    ]

    for img_name in target_images:
        img_path = os.path.join(current_dir, img_name)
        if os.path.exists(img_path):
            for mode in modes:
                adjust_brightness_by_qr(
                    img_path,
                    qr_path,
                    mode["bright_decrease"],
                    clip_margin=PRE_CLIP_MARGIN,
                    invert=mode["invert"],
                    output_suffix=f"{abs(int(mode['bright_decrease']))}_{mode['suffix']}"
                )
        else:
            print(f"Warning: {img_name} が見つかりません")

if __name__ == "__main__":
    main()