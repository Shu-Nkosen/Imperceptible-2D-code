import cv2
import numpy as np

def detect_display_frame(input_file, edge_output, line_output, contour_output, final_output):
    """
    ディスプレイの枠を検出し、各処理段階の画像を保存
    
    Parameters:
        input_file (str): 入力画像のファイル名
        edge_output (str): エッジ検出後の画像
        line_output (str): 直線検出後の画像
        contour_output (str): 矩形候補の検出後の画像
        final_output (str): 最適な矩形を選択した後の画像
    """

    # 画像を読み込み
    image = cv2.imread(input_file)
    if image is None:
        print(f"エラー: 画像 '{input_file}' を読み込めません。")
        return
    
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ノイズ除去（ガウシアンブラー）
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny エッジ検出
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.bitwise_not(edges)

    # 黒い縁を追加
    border_size = 10  # 縁の幅（ピクセル単位）
    edges = cv2.copyMakeBorder(edges, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0)
    cv2.imwrite(edge_output, edges)  # 🟢 エッジ検出結果を保存


    # エッジ補強（膨張処理）
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # ハフ変換で直線検出
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    
    # 直線を描画
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 直線を重ねた画像を保存
    combined_lines = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    cv2.imwrite(line_output, combined_lines)  # 🟢 直線検出結果を保存

    # 輪郭検出（矩形候補を見つける）
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 矩形候補を描画
    contour_image = image.copy()
    for contour in contours:
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:  # 4点（矩形）のみ描画
            cv2.drawContours(contour_image, [approx], -1, (255, 0, 0), 2)

    cv2.imwrite(contour_output, contour_image)  # 🟢 矩形候補の描画結果を保存

    # 画像の中心座標
    img_height, img_width = image.shape[:2]
    image_center = (img_width // 2, img_height // 2)

    # 矩形候補をフィルタリング
    best_rectangle = None
    min_center_distance = float('inf')

    for contour in contours:
        # 輪郭の近似（四角形を見つける）
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:  # 4点（矩形）
            # 矩形の中心を求める
            rect_center = np.mean(approx, axis=0)[0]
            center_distance = np.linalg.norm(np.array(rect_center) - np.array(image_center))

            # 画像の中心に最も近い矩形を選択
            if center_distance < min_center_distance:
                min_center_distance = center_distance
                best_rectangle = approx

    # 最適な矩形を描画
    final_image = contour_image.copy()
    if best_rectangle is not None:
        cv2.drawContours(final_image, [best_rectangle], -1, (0, 0, 255), 2)

    # 最終結果を保存
    cv2.imwrite(final_output, final_image)
    print(f"✅ 各段階の検出結果を保存しました。")

# === ここから実行部分 ===
if __name__ == "__main__":
    file = "15"  # 画像のファイル名（番号）
    input_file = f"{file}.jpg"  # 入力画像
    edge_output = f"{file}_edges.jpg"  # エッジ検出結果
    line_output = f"{file}_lines.jpg"  # 直線検出結果
    contour_output = f"{file}_contours.jpg"  # 矩形候補
    final_output = f"{file}_final.jpg"  # 最適な矩形を選択後

    detect_display_frame(input_file, edge_output, line_output, contour_output, final_output)
