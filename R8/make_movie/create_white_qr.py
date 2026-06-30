import cv2
import numpy as np
import os

INVERT_QR = False  # True: QRコードの白黒を反転, False: 通常


def place_qr_on_white(image_path: str, qr_path: str, invert: bool = False, output_tag: str | None = None) -> None:
    image = cv2.imread(image_path)
    qr_code = cv2.imread(qr_path, cv2.IMREAD_GRAYSCALE)
    if image is None or qr_code is None:
        print(f"Error: 画像の読み込みに失敗しました - {image_path} or {qr_path}")
        return

    img_h, img_w = image.shape[:2]
    square_size = img_h
    x_offset = (img_w - square_size) // 2

    qr_resized = cv2.resize(qr_code, (square_size, square_size))
    _, qr_binary = cv2.threshold(qr_resized, 127, 255, cv2.THRESH_BINARY)
    if invert:
        qr_binary = 255 - qr_binary

    # create a white canvas matching the original resolution
    canvas = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
    canvas[:, x_offset:x_offset + square_size, 0] = qr_binary
    canvas[:, x_offset:x_offset + square_size, 1] = qr_binary
    canvas[:, x_offset:x_offset + square_size, 2] = qr_binary

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    invert_str = "inv" if invert else "normal"
    if output_tag is None:
        output_tag = f"white_qr_{invert_str}"
    output_name = f"{base_name}_{output_tag}.png"
    output_path = os.path.join(os.path.dirname(image_path), output_name)

    cv2.imwrite(output_path, canvas)
    print(f"保存完了: {output_name}")


def main() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    qr_path = os.path.join(current_dir, "HP_QR.png")
    target_images = ["rice.png", "kosen.png", "nagaoka_fireworks.png", "hocho.png", "ex.png"]

    for img_name in target_images:
        img_path = os.path.join(current_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: {img_name} が見つかりません")
            continue
        place_qr_on_white(img_path, qr_path, invert=INVERT_QR)


if __name__ == "__main__":
    main()
