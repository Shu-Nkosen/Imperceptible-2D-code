import cv2
import numpy as np
import os

# ======= パラメータ設定 =======
XX = 8
# 変化させるRGB量
BRIGHTNESS_DECREASE_POS = XX  # 暗くする量
BRIGHTNESS_DECREASE_NEG = -XX # 明るくする量
PRE_CLIP_MARGIN = 4           # 事前クリップマージン
# ==============================

def adjust_brightness_by_qr(image_path, qr_path, bright_decrease, clip_margin, value_mode="max", output_tag=None):
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

    # float32に変換
    square_region = image[:, x_offset:x_offset+square_size, :].astype(np.float32)

    # =========================================================================
    # 事前クリッピング
    # =========================================================================
    if clip_margin > 0:
        square_region = np.clip(square_region, clip_margin, 255.0 - clip_margin)

    # マスクを作成
    black_mask = qr_binary == 0

    # =========================================================================
    # 計算ロジック (Max/Min 完全対称)
    # =========================================================================
    
    target_indices = None

    if value_mode == "min":
        # I系 (Minモード): 最小チャンネルを選択
        target_indices = np.argmin(square_region, axis=2)
    else:
        # X系 (Maxモード): 最大チャンネルを選択
        # ※以前のスケール計算を廃止し、argmin同様にインデックスを取得
        target_indices = np.argmax(square_region, axis=2)

    # 選択された1つのチャンネルに対してのみ値を変更
    for c in range(3):
        # 「QRの黒部分」かつ「このチャンネル(c)が対象(最大or最小)である画素」を特定
        target_mask = black_mask & (target_indices == c)
        
        # 対象箇所のみ値を変更
        # bright_decreaseが正なら暗く(-)、負なら明るく(-- = +)なる
        square_region[:, :, c][target_mask] -= bright_decrease

    # =========================================================================

    # 最終クリッピング
    square_region = np.clip(square_region, 0, 255)

    # 書き戻し
    result_img = image.astype(np.float32)
    result_img[:, x_offset:x_offset+square_size, :] = square_region
    result = result_img.astype(np.uint8)
    
    # 保存
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    if output_tag is None:
        output_tag = f"{abs(int(bright_decrease))}_{value_mode}"
    output_name = f"{base_name}_{output_tag}.png"
    output_path = os.path.join(os.path.dirname(image_path), output_name)

    cv2.imwrite(output_path, result)
    print(f"保存完了: {output_name} (Mode: {value_mode}, Margin: {clip_margin})")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    qr_path = os.path.join(current_dir, "HP_QR.png")
    target_images = ["rice.png", "kosen.png", "nagaoka_fireworks.png", "hocho.png", "ex.png"]

    modes = [
        {
            "bright_decrease": BRIGHTNESS_DECREASE_POS,
            "value_mode": "max",
            "suffix": "normalX", # Maxチャンネルのみ操作
        },
        {
            "bright_decrease": BRIGHTNESS_DECREASE_NEG,
            "value_mode": "max",
            "suffix": "invX",    # Maxチャンネルのみ操作
        },
        {
            "bright_decrease": BRIGHTNESS_DECREASE_POS,
            "value_mode": "min",
            "suffix": "normalI", # Minチャンネルのみ操作
        },
        {
            "bright_decrease": BRIGHTNESS_DECREASE_NEG,
            "value_mode": "min",
            "suffix": "invI",    # Minチャンネルのみ操作
        },
    ]

    for img_name in target_images:
        img_path = os.path.join(current_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Skip: {img_name}")
            continue
            
        for mode in modes:
            output_tag = f"{abs(int(mode['bright_decrease']))}_{mode['suffix']}"
            adjust_brightness_by_qr(
                image_path=img_path,
                qr_path=qr_path,
                bright_decrease=mode["bright_decrease"],
                clip_margin=PRE_CLIP_MARGIN,
                value_mode=mode["value_mode"],
                output_tag=output_tag,
            )

if __name__ == "__main__":
    main()