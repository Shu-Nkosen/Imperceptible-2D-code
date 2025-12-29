import cv2
import numpy as np
import os

# パラメータ設定
XX = 4
BRIGHTNESS_DECREASE_POS = XX  # 黒部分の輝度減少量（normal）
BRIGHTNESS_DECREASE_NEG = -XX  # 黒部分の輝度増加量（inv）

def adjust_brightness_by_qr(image_path, qr_path, bright_decrease, invert=False, output_suffix=""):
    """
    QRコードのパターンに基づいて画像の輝度を調整
    
    Args:
        image_path: 元画像のパス
        qr_path: QRコードのパス
        bright_increase: 白部分の輝度増加量
        bright_decrease: 黒部分の輝度減少量
        invert: QRコードの白黒を反転するか
    """
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
    

    if invert:
        qr_binary = 255 - qr_binary

    max_change = abs(bright_decrease)
    if max_change > 0:
        lower_bound = float(max_change)
        upper_bound = float(255 - max_change)
        if lower_bound >= upper_bound:
            print(f"Error: 変化量が大きすぎます (bright_decrease={bright_decrease})")
            return
    else:
        lower_bound, upper_bound = 0.0, 255.0

    # マスクを作成 (白部分と黒部分)
    white_mask = qr_binary == 255
    black_mask = qr_binary == 0

    # RGB各チャネルごとに出力
    channel_info = [("R", 2), ("G", 1), ("B", 0)]
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    invert_str = "inv" if invert else "normal"


    for channel_name, channel_idx in channel_info:
        channel_image = image.astype(np.float32)

        if max_change > 0:
            channel_image[:, :, channel_idx] = np.clip(channel_image[:, :, channel_idx], lower_bound, upper_bound)

        channel_region = channel_image[:, x_offset:x_offset + square_size, channel_idx]
        channel_region[black_mask] -= bright_decrease
        channel_region = np.clip(channel_region, 0, 255)
        channel_image[:, x_offset:x_offset + square_size, channel_idx] = channel_region

        result = channel_image.astype(np.uint8)
        tag = output_suffix or f"{abs(int(bright_decrease))}_{invert_str}"
        output_name = f"{base_name}_{tag}{channel_name}.png"
        output_path = os.path.join(os.path.dirname(image_path), output_name)

        cv2.imwrite(output_path, result)
        print(f"保存完了: {output_name}")


def main():
    # カレントディレクトリを取得
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # QRコードのパス
    qr_path = os.path.join(current_dir, "HP_QR.png")
    
    # 処理対象の画像リスト (QR以外)
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

    # 各画像を処理（normal/inv の2パターン × RGB）
    for img_name in target_images:
        img_path = os.path.join(current_dir, img_name)
        if os.path.exists(img_path):
            for mode in modes:
                adjust_brightness_by_qr(
                    img_path,
                    qr_path,
                    mode["bright_decrease"],
                    invert=mode["invert"],
                    output_suffix=f"{abs(int(mode['bright_decrease']))}_{mode['suffix']}"
                )
        else:
            print(f"Warning: {img_name} が見つかりません")

if __name__ == "__main__":
    main()