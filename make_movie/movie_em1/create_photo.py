import cv2
import numpy as np
import os

# パラメータ設定
XX = 10
BRIGHTNESS_DECREASE_POS = XX
BRIGHTNESS_DECREASE_NEG = -XX

def adjust_brightness_by_qr(image_path, qr_path, bright_decrease, value_mode="max", output_tag=None):
    image = cv2.imread(image_path)
    qr_code = cv2.imread(qr_path, cv2.IMREAD_GRAYSCALE)
    if image is None or qr_code is None:
        print(f"Error: 画像の読み込みに失敗しました - {image_path} or {qr_path}")
        return
    
    # 画像のサイズを取得
    img_h, img_w = image.shape[:2]
    
    # 正方形領域のサイズを計算（高さに合わせる）
    square_size = img_h
    
    # 画像の中央に正方形領域を配置するためのオフセット
    x_offset = (img_w - square_size) // 2
    
    # QRコードを正方形サイズにリサイズ
    qr_resized = cv2.resize(qr_code, (square_size, square_size))

    # QRコードを2値化 (白=255, 黒=0)
    _, qr_binary = cv2.threshold(qr_resized, 127, 255, cv2.THRESH_BINARY)

    # 正方形領域のみを処理
    square_region = image[:, x_offset:x_offset+square_size, :].astype(np.float32)

    max_change = abs(bright_decrease)
    if max_change > 0:
        lower_bound = float(max_change)
        upper_bound = float(255 - max_change)
        if lower_bound >= upper_bound:
            print(f"Error: 変化量が大きすぎます (bright_decrease={bright_decrease})")
            return
        # 事前クリップしてオーバーフローを防止
        square_region = np.clip(square_region, lower_bound, upper_bound)

    # マスクを作成 (白部分と黒部分)
    white_mask = qr_binary == 255
    black_mask = qr_binary == 0

    # V相当を手計算で取得
    if value_mode == "min":
        v_channel = square_region.min(axis=2)
    else:
        v_channel = square_region.max(axis=2)

    # Vを加減算
    v_new = v_channel.copy()
    v_new[black_mask] -= bright_decrease
    v_new = np.clip(v_new, 0, 255)

    # チャンネルを比率でスケールして明るさを反映
    denom = np.where(v_channel > 0, v_channel, 1.0)
    scale = v_new / denom
    for c in range(3):
        square_region[:, :, c] = np.clip(square_region[:, :, c] * scale, 0, 255)

    # 処理した正方形領域を元の画像に戻す
    result_img = image.astype(np.float32)
    result_img[:, x_offset:x_offset+square_size, :] = square_region
    result = result_img.astype(np.uint8)
    
    # ファイル名を生成
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    if output_tag is None:
        output_tag = f"{abs(int(bright_decrease))}_{value_mode}"
    output_name = f"{base_name}_{output_tag}.png"
    output_path = os.path.join(os.path.dirname(image_path), output_name)

    # 保存
    cv2.imwrite(output_path, result)
    print(f"保存完了: {output_name}")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    qr_path = os.path.join(current_dir, "HP_QR.png")
    target_images = ["rice.png", "kosen.png", "nagaoka_fireworks.png", "hocho.png", "ex.png"]

    modes = [
        {
            "bright_decrease": BRIGHTNESS_DECREASE_POS,      # max V, black darken
            "value_mode": "max",
            "suffix": "normalX",
        },
        {
            "bright_decrease": BRIGHTNESS_DECREASE_NEG,      # max V, black brighten
            "value_mode": "max",
            "suffix": "invX",
        },
        {
            "bright_decrease": BRIGHTNESS_DECREASE_POS,      # min V, black darken
            "value_mode": "min",
            "suffix": "normalI",
        },
        {
            "bright_decrease": BRIGHTNESS_DECREASE_NEG,      # min V, black brighten
            "value_mode": "min",
            "suffix": "invI",
        },
    ]


    for img_name in target_images:
        img_path = os.path.join(current_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: {img_name} が見つかりません")
            continue
        for mode in modes:
            output_tag = f"{abs(int(mode['bright_decrease']))}_{mode['suffix']}"
            adjust_brightness_by_qr(
                img_path,
                qr_path,
                mode["bright_decrease"],
                value_mode=mode["value_mode"],
                output_tag=output_tag,
            )

if __name__ == "__main__":
    main()