import cv2
import numpy as np
import os

# パラメータ設定
BRIGHTNESS_INCREASE = 1  # 白部分の輝度増加量
BRIGHTNESS_DECREASE = 1  # 黒部分の輝度減少量
INVERT_QR = False  # True: QRコードの白黒を反転, False: 通常

def adjust_brightness_by_qr(image_path, qr_path, bright_increase, bright_decrease, invert=False):
    """
    QRコードのパターンに基づいて画像の輝度を調整
    
    Args:
        image_path: 元画像のパス
        qr_path: QRコードのパス
        bright_increase: 白部分の輝度増加量
        bright_decrease: 黒部分の輝度減少量
        invert: QRコードの白黒を反転するか
    """
    # 画像とQRコードを読み込み
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
    invert_str = "inv" if invert else "normal"
    output_name = f"{base_name}_b{bright_increase}_d{bright_decrease}_{invert_str}.png"
    output_path = os.path.join(os.path.dirname(image_path), output_name)
    
    # 保存
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
        "hocho.png"
    ]
    
    # 各画像を処理
    for img_name in target_images:
        img_path = os.path.join(current_dir, img_name)
        if os.path.exists(img_path):
            adjust_brightness_by_qr(
                img_path, 
                qr_path, 
                BRIGHTNESS_INCREASE, 
                BRIGHTNESS_DECREASE, 
                INVERT_QR
            )
        else:
            print(f"Warning: {img_name} が見つかりません")

if __name__ == "__main__":
    main()