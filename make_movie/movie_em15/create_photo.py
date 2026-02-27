import cv2
import numpy as np
import os

# ======= パラメータ設定 =======
XX = 100
# 変化させるRGB量
BRIGHTNESS_DECREASE_POS = XX  # 暗くする量
BRIGHTNESS_DECREASE_NEG = -XX # 明るくする量
PRE_CLIP_MARGIN = 4           # 事前クリップマージン
# ==============================

def prepare_processing_context(image_path, qr_path, clip_margin):
    image = cv2.imread(image_path)
    qr_code = cv2.imread(qr_path, cv2.IMREAD_GRAYSCALE)
    if image is None or qr_code is None:
        print(f"Error: 画像の読み込みに失敗しました - {image_path} or {qr_path}")
        return None

    # 入力画像をフルHD(1920x1080)に揃えて処理
    image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_AREA)

    img_h, img_w = image.shape[:2]
    square_size = img_h
    x_offset = (img_w - square_size) // 2

    qr_resized = cv2.resize(qr_code, (square_size, square_size))
    _, qr_binary = cv2.threshold(qr_resized, 127, 255, cv2.THRESH_BINARY)
    black_mask = qr_binary == 0

    square_region = image[:, x_offset:x_offset+square_size, :].astype(np.float32)
    if clip_margin > 0:
        square_region = np.clip(square_region, clip_margin, 255.0 - clip_margin)

    # 同一パラメータのnormal/invで同じチャネルを再利用できるようにインデックスをキャッシュ
    target_indices = {
        "max": np.argmax(square_region, axis=2),
        "min": np.argmin(square_region, axis=2),
    }

    return {
        "image": image,
        "square_region": square_region.copy(),
        "black_mask": black_mask,
        "target_indices": target_indices,
        "x_offset": x_offset,
        "square_size": square_size,
        "base_dir": os.path.dirname(image_path),
        "base_name": os.path.splitext(os.path.basename(image_path))[0],
        "clip_margin": clip_margin,
    }


def save_adjusted_variant(context, bright_decrease, value_mode, output_tag):
    target_indices = context["target_indices"].get(value_mode)
    if target_indices is None:
        print(f"Skip: 未対応モード {value_mode}")
        return

    square_region = context["square_region"].copy()
    black_mask = context["black_mask"]

    for c in range(3):
        target_mask = black_mask & (target_indices == c)
        if target_mask.any():
            square_region[:, :, c][target_mask] -= bright_decrease

    square_region = np.clip(square_region, 0, 255)

    result_img = context["image"].astype(np.float32).copy()
    x_offset = context["x_offset"]
    square_size = context["square_size"]
    result_img[:, x_offset:x_offset+square_size, :] = square_region
    result = result_img.astype(np.uint8)

    output_name = f"{context['base_name']}_{output_tag}.png"
    output_path = os.path.join(context["base_dir"], output_name)

    cv2.imwrite(output_path, result)
    print(f"保存完了: {output_name} (Mode: {value_mode}, Margin: {context['clip_margin']})")


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
            
        context = prepare_processing_context(img_path, qr_path, PRE_CLIP_MARGIN)
        if context is None:
            continue

        for mode in modes:
            output_tag = f"{abs(int(mode['bright_decrease']))}_{mode['suffix']}"
            save_adjusted_variant(
                context=context,
                bright_decrease=mode["bright_decrease"],
                value_mode=mode["value_mode"],
                output_tag=output_tag,
            )

if __name__ == "__main__":
    main()