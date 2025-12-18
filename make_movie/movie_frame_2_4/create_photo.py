import cv2
import numpy as np
import os

# パラメータ設定
XX = 2
BRIGHTNESS_INCREASE_MAX = 0
BRIGHTNESS_DECREASE_MAX = XX
BRIGHTNESS_INCREASE_MIN = 0
BRIGHTNESS_DECREASE_MIN = XX
INVERT_QR = True  # True: QRコードの白黒を反転, False: 通常

def adjust_brightness_by_qr(image_path, qr_path, bright_increase, bright_decrease, invert=False, output_tag=None):
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
    
    # 反転オプション
    if invert:
        qr_binary = 255 - qr_binary
    
    # BGRをHSVに変換
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # 変化量の最大値で事前クリップ
    max_change = max(bright_increase, bright_decrease)
    if max_change > 0:
        lower_bound = float(max_change)
        upper_bound = float(255 - max_change)
        if lower_bound >= upper_bound:
            print(f"Error: 変化量が大きすぎます (bright_increase={bright_increase}, bright_decrease={bright_decrease})")
            return
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], lower_bound, upper_bound)

    # 正方形領域のみを処理
    square_region = hsv[:, x_offset:x_offset+square_size, :]
    
    # マスクを作成 (白部分と黒部分)
    white_mask = qr_binary == 255
    black_mask = qr_binary == 0
    
    # 輝度(V)チャンネルを調整
    square_region[:, :, 2][white_mask] += bright_increase
    square_region[:, :, 2][black_mask] -= bright_decrease
    
    # 輝度を0-255の範囲にクリップ
    square_region[:, :, 2] = np.clip(square_region[:, :, 2], 0, 255)
    
    # 処理した正方形領域を元の画像に戻す
    hsv[:, x_offset:x_offset+square_size, :] = square_region
    
    # HSVをBGRに戻す
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # ファイル名を生成
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    if output_tag is None:
        invert_str = "inv" if invert else "normal"
        output_tag = f"b{bright_increase}_d{bright_decrease}_{invert_str}"
    output_name = f"{base_name}_{output_tag}.png"
    output_path = os.path.join(os.path.dirname(image_path), output_name)

    # 保存
    cv2.imwrite(output_path, result)
    print(f"保存完了: {output_name}")

def compute_min_channel_difference(img1: np.ndarray, img2: np.ndarray):
    diff_all = img2 - img1
    abs_diff = np.abs(diff_all)
    min_indices = np.argmin(abs_diff, axis=2)
    diff_map = np.take_along_axis(diff_all, min_indices[..., None], axis=2).squeeze(-1)
    return diff_map



def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    qr_path = os.path.join(current_dir, "HP_QR.png")
    target_images = ["rice.png", "kosen.png", "nagaoka_fireworks.png", "hocho.png", "ex.png"]

    modes = [
        {
            "bright_increase": BRIGHTNESS_INCREASE_MAX,
            "bright_decrease": BRIGHTNESS_DECREASE_MAX,
            "invert": True,
            "suffix": "X",
        },
        {
            "bright_increase": BRIGHTNESS_INCREASE_MAX,
            "bright_decrease": BRIGHTNESS_DECREASE_MAX,
            "invert": False,
            "suffix": "X",
        },
        {
            "bright_increase": BRIGHTNESS_INCREASE_MIN,
            "bright_decrease": BRIGHTNESS_DECREASE_MIN,
            "invert": True,
            "suffix": "I",
        },
        {
            "bright_increase": BRIGHTNESS_INCREASE_MIN,
            "bright_decrease": BRIGHTNESS_DECREASE_MIN,
            "invert": False,
            "suffix": "I",
        },
    ]


    for img_name in target_images:
        img_path = os.path.join(current_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: {img_name} が見つかりません")
            continue
        for mode in modes:
            invert_str = "inv" if mode["invert"] else "normal"
            output_tag = f"b{mode['bright_increase']}_d{mode['bright_decrease']}_{invert_str}{mode['suffix']}"
            adjust_brightness_by_qr(
                img_path,
                qr_path,
                mode["bright_increase"],
                mode["bright_decrease"],
                invert=mode["invert"],
                output_tag=output_tag,
            )

if __name__ == "__main__":
    main()