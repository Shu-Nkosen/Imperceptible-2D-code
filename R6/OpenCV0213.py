import cv2
import numpy as np
from itertools import combinations
import math

# 既存の直線との距離が近いかどうかを確認する関数
def is_similar_line(line1, line2, threshold):
    """
    2つの直線の中点の距離がthreshold以下であれば類似していると判定
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # 中点を計算
    mid1 = ((x1 + x2) / 2, (y1 + y2) / 2)
    mid2 = ((x3 + x4) / 2, (y3 + y4) / 2)

    # 中点間の距離を計算
    distance = np.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)
    return distance <= threshold

# 近すぎる直線を除外してからペアを作成
def filter_pairs(lines, min_dist):
    valid_pairs = []
    for (line1, line2) in combinations(lines, 2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        mid1 = ((x1 + x2) / 2, (y1 + y2) / 2)
        mid2 = ((x3 + x4) / 2, (y3 + y4) / 2)
        dist = np.sqrt((mid1[0] - mid2[0]) ** 2 + (mid1[1] - mid2[1]) ** 2)
        
        # 閾値以上の距離のペアのみ追加
        if dist >= min_dist:
            valid_pairs.append((line1, line2))
    return valid_pairs[:2000]  # ペアの数を制限してパフォーマンス改善

# 枠を探す
def check_frame(sameLineThreshold):
    # 水平線・垂直線のリスト
    horizontal_lines = []
    vertical_lines = []

    # 直線を分類（類似する直線を除外）
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angle = abs(angle)

            # 水平線・垂直線の判定
            if angle < 10 or angle > 170:  # 水平線
                if not any(is_similar_line(line[0], existing_line, sameLineThreshold) for existing_line in horizontal_lines):
                    horizontal_lines.append(line[0])
            elif 80 < angle < 100:  # 垂直線
                if not any(is_similar_line(line[0], existing_line, sameLineThreshold) for existing_line in vertical_lines):
                    vertical_lines.append(line[0])


    horizontal_pairs = filter_pairs(horizontal_lines, min_height_dist)
    vertical_pairs = filter_pairs(vertical_lines, min_width_dist)

    # 条件を満たす矩形を探す
    rectangles = []  # 見つかった矩形を保存
    for h_line1, h_line2 in horizontal_pairs:
        for v_line1, v_line2 in vertical_pairs:
            top_left = get_intersection(h_line1, v_line1)
            top_right = get_intersection(h_line1, v_line2)
            bottom_left = get_intersection(h_line2, v_line1)
            bottom_right = get_intersection(h_line2, v_line2)

            if None not in (top_left, top_right, bottom_left, bottom_right):
                # アスペクト比を計算
                width = np.linalg.norm(np.array(top_right) - np.array(top_left))
                height = np.linalg.norm(np.array(bottom_left) - np.array(top_left))
                aspect_ratio = width / height if height > 0 else 0

                # アスペクト比が条件を満たす場合のみ追加
                if aspect_target - aspect_tolerance <= aspect_ratio <= aspect_target + aspect_tolerance:
                    rectangles.append((top_left, top_right, bottom_right, bottom_left))

    return rectangles

# 交点を求める関数
def get_intersection(line1, line2):
    x1, y1, x2, y2 = map(float, line1)
    x3, y3, x4, y4 = map(float, line2)
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) > 1e-4:  # 分母がほぼゼロでないことを確認
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        
        # 座標のバリデーション
        if 0 <= px <= img_width and 0 <= py <= img_height:
            return int(px), int(py)
    return None

# 平面補正して切り出す関数
def warp_perspective(image, rectangle, output_filename):
    # 頂点を取得
    top_left, top_right, bottom_right, bottom_left = rectangle

    # 頂点の順序が正しいか確認（必要であれば修正）
    points = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    
    # 左上を基準にソート
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    ordered_points = np.zeros_like(points)
    ordered_points[0] = points[np.argmin(s)]  # 左上
    ordered_points[2] = points[np.argmax(s)]  # 右下
    ordered_points[1] = points[np.argmin(diff)]  # 右上
    ordered_points[3] = points[np.argmax(diff)]  # 左下

    # 変換後の画像の幅と高さを計算
    width = int(np.linalg.norm(ordered_points[1] - ordered_points[0]))
    height = int(np.linalg.norm(ordered_points[0] - ordered_points[3]))

    # 変換後の画像の頂点
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # 透視変換行列を計算
    transform_matrix = cv2.getPerspectiveTransform(ordered_points, dst_points)

    # 透視変換を適用
    warped_image = cv2.warpPerspective(image, transform_matrix, (width, height))

    # 結果を保存
    # cv2.imwrite(output_filename, warped_image)

    return warped_image


def calculate_aspect_ratio(rectangle):

    top_left, top_right, bottom_right, bottom_left = rectangle

    # 幅と高さを計算
    width = abs(top_right[0] - top_left[0])
    height = abs(bottom_left[1] - top_left[1])

    # アスペクト比を計算
    aspect_ratio = width / height if height > 0 else 0
    return aspect_ratio

def is_center_close(rectangle, image_center, max_center_offset_x, max_center_offset_y):
    top_left, top_right, bottom_right, bottom_left = rectangle
    rectangle_center = (
        (top_left[0] + top_right[0] + bottom_left[0] + bottom_right[0]) / 4,
        (top_left[1] + top_right[1] + bottom_left[1] + bottom_right[1]) / 4
    )
    distance_x = abs(rectangle_center[0] - image_center[0])
    distance_y = abs(rectangle_center[1] - image_center[1])

        # デバッグ出力
    # if distance_x <= max_center_offset_x and distance_y <= max_center_offset_y:
        # print(f"矩形中心: {rectangle_center}, 画像中心: {image_center}, 横距離: {distance_x}, 縦距離: {distance_y}")
        # print(f"許容範囲X: {max_center_offset_x}, 許容範囲Y: {max_center_offset_y}")

    # 判定
    return distance_x <= max_center_offset_x and distance_y <= max_center_offset_y

def calculate_rectangle_area(rectangle):
    top_left, top_right, bottom_right, bottom_left = rectangle
    width = np.linalg.norm(np.array(top_right) - np.array(top_left))
    height = np.linalg.norm(np.array(top_left) - np.array(bottom_left))
    return width * height

def wright_red_line():
        # デバッグ用にすべての矩形を赤い線で描画
    for rectangle in rectangles:
        top_left, top_right, bottom_right, bottom_left = rectangle
        cv2.line(debug_image, top_left, top_right, (0, 0, 255), 2)
        cv2.line(debug_image, top_right, bottom_right, (0, 0, 255), 2)
        cv2.line(debug_image, bottom_right, bottom_left, (0, 0, 255), 2)
        cv2.line(debug_image, bottom_left, top_left, (0, 0, 255), 2)

# 最適な矩形を選択する関数
def find_best_rectangle(rectangles, target_aspect, image_center, max_center_offset_x, max_center_offset_y, areaSize):
    best_rectangle = None
    min_aspect_diff = float('inf')
    image_area = img_width * img_height  # 入力画像の面積
    ok = 0

    for rectangle in rectangles:
        aspect_ratio = calculate_aspect_ratio(rectangle)

            # 中心条件を満たすか確認
        if not is_center_close(rectangle, image_center, max_center_offset_x, max_center_offset_y):
            continue
            # 面積条件を満たすか確認
        rectangle_area = calculate_rectangle_area(rectangle)
        if rectangle_area < image_area / areaSize:
            continue
        
        ok += 1
        print (f"{ok}")

        # アスペクト比がターゲットに最も近い矩形を選択
        aspect_diff = abs(aspect_ratio - target_aspect)
        if aspect_diff < min_aspect_diff:
            min_aspect_diff = aspect_diff
            best_rectangle = rectangle

    return best_rectangle

# 画像を取得する関数
def get_image():
    # 画像の読み込み
    image = cv2.imread(input)
    if image is None:
        raise ValueError(f"画像 '{input}' が読み込めませんでした。")
    
    # デバッグ用のコピー
    debug_image = image.copy()
    
    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ノイズ除去（ガウシアンブラー）
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny エッジ検出
    edges = cv2.Canny(blurred, 50, 150)
    # cv2.imwrite(edge_output, edges)  # 🟢 エッジ検出結果を保存

    # エッジ補強（膨張処理）
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # ハフ変換で直線検出
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # 画像の幅と高さを取得
    img_height, img_width = image.shape[:2]
    min_width_dist = img_width / areaSize
    min_height_dist = img_height / areaSize

    # 元画像の中心を計算
    image_center = (img_width / 2, img_height / 2)
    max_center_offset_x = img_width * doubleX  # 横方向の許容範囲
    max_center_offset_y = img_height * doubleY  # 縦方向の許容範囲

    
    return image, debug_image, gray, edges, lines, img_height, img_width, min_width_dist, min_height_dist, image_center, max_center_offset_x, max_center_offset_y


# 黒い淵を削除
def cutting(image, threshold, border_width, black_ratio):
    """
    指定された範囲すべてを調査し、進むごとに黒い割合が閾値を超える場合、切り取る範囲を更新。

    Parameters:
        image (ndarray): 入力画像
        threshold (int): 二値化の閾値
        border_width (int): 外側のチェック範囲の幅
        black_ratio (float): 黒いピクセルが占める割合の閾値 (0〜1)

    Returns:
        tuple: (トリミング後の画像, グレースケール画像, 二値化画像, カット情報)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 二値化
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    h, w = binary.shape

    # トリミング位置を記録
    top, bottom, left, right = 0, h, 0, w

    # トリミング情報を保持
    cut_info = {"top": [], "bottom": [], "left": [], "right": []}

    # 上辺のチェック
    for i in range(border_width):
        black_ratio_top = np.sum(binary[i:i+1, :] == 0) / binary[i:i+1, :].size
        # print(f"Top border check at row {i}: black_ratio = {black_ratio_top}")
        if black_ratio_top >= black_ratio:
            top = i + 1
            cut_info["top"].append(f"Cut updated at row {i} due to high black_ratio ({black_ratio_top})")

    # 下辺のチェック
    for i in range(1, border_width + 1):
        black_ratio_bottom = np.sum(binary[-i:, :] == 0) / binary[-i:, :].size
        # print(f"Bottom border check at row {-i}: black_ratio = {black_ratio_bottom}")
        if black_ratio_bottom >= black_ratio:
            bottom = h - i
            cut_info["bottom"].append(f"Cut updated at row {-i} due to high black_ratio ({black_ratio_bottom})")

    # 左辺のチェック
    for i in range(border_width):
        black_ratio_left = np.sum(binary[:, i:i+1] == 0) / binary[:, i:i+1].size
        # print(f"Left border check at column {i}: black_ratio = {black_ratio_left}")
        if black_ratio_left >= black_ratio:
            left = i + 1
            cut_info["left"].append(f"Cut updated at column {i} due to high black_ratio ({black_ratio_left})")

    # 右辺のチェック
    for i in range(1, border_width + 1):
        black_ratio_right = np.sum(binary[:, -i:] == 0) / binary[:, -i:].size
        # print(f"Right border check at column {-i}: black_ratio = {black_ratio_right}")
        if black_ratio_right >= black_ratio:
            right = w - i
            cut_info["right"].append(f"Cut updated at column {-i} due to high black_ratio ({black_ratio_right})")

    # トリミング
    cropped_image = image[top:bottom, left:right]
    return cropped_image, gray, binary, cut_info

# Boat
# Lenna
# Music
# News

picture = 0

# position = "C"

picture_name = ""

setting = 5

sameLineThreshold = 400

aspect_target = 1.778  # 16:9のアスペクト比
aspect_tolerance = 0.15

doubleX = 0.02
doubleY = 0.02

areaSize = 2

# 切り取り条件
threshold = 50
border_width = 100
black_ratio = 0.8


for i in range(1, 41 , 1):
    s =int ((i-1) /4 +1) 
    picture = s*5
    input = f"{i}.jpg"
    if i%4 == 1:
        picture_name = "Boat"
    elif i%4 == 2:
        picture_name = "Lenna"
    elif i%4 == 3:
        picture_name = "Music"
    elif i%4 == 0:
        picture_name = "News"
        

    # picture = math.floor((picture+3) / 4) * 5
    output = f"{picture_name}{picture}.png"
    

    # 実行

    image, debug_image, gray, edges, lines, img_height, img_width, min_width_dist, min_height_dist, image_center, max_center_offset_x, max_center_offset_y = get_image()

    # デバッグ画像を保存
    cv2.imwrite("debug_rectangles.png", debug_image)

    rectangles = check_frame(sameLineThreshold)
    best = 0
    while best == 0:
        # 条件を満たす矩形がある場合
        if rectangles:
            best_rectangle = find_best_rectangle(rectangles, aspect_target, image_center, max_center_offset_x, max_center_offset_y,areaSize)
            wright_red_line()
            if best_rectangle:
                lastImage = warp_perspective(image, best_rectangle, output)
                best = 1
            else:
                print("条件に一致する矩形は見つかりませんでした。")
                aspect_tolerance += 0.03
                doubleX += 0.01
                doubleY += 0.01
                # areaSize += 0.5

                # 実行
                image, debug_image, gray, edges, lines, img_height, img_width, min_width_dist, min_height_dist, image_center, max_center_offset_x, max_center_offset_y = get_image()

                # デバッグ画像を保存
                cv2.imwrite("debug_rectangles.png", debug_image)

                rectangles = check_frame(sameLineThreshold)
        else:
            print("矩形が検出されませんでした。")

            aspect_tolerance += 0.02
            doubleX += 0.02
            doubleY += 0.02
            areaSize += 0.5

            # 実行
            image, debug_image, gray, edges, lines, img_height, img_width, min_width_dist, min_height_dist, image_center, max_center_offset_x, max_center_offset_y = get_image()


            rectangles = check_frame(sameLineThreshold)

    result, gray_image, binary_image, cut_info = cutting(lastImage, threshold, border_width, black_ratio)


    # 結果を保存
    cv2.imwrite(f'{output}', result)
    print(f"最適な矩形を保存しました: {output}")
