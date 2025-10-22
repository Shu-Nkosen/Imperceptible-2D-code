import cv2
import numpy as np

# グローバル変数
points = []  # クリックされた点を格納
window_name = "Select 4 Points"
resize_factor = 0.5  # 画像のリサイズ率（0.5は50%縮小）

def mouse_callback(event, x, y, flags, param):
    """
    マウス操作のコールバック関数
    左クリックで座標を4点指定
    """
    global points
    if event == cv2.EVENT_LBUTTONDOWN:  # 左クリックで点を選択
        # クリック座標を元画像の座標にスケーリング
        real_x = int(x / resize_factor)
        real_y = int(y / resize_factor)
        points.append((real_x, real_y))
        print(f"Point selected: ({real_x}, {real_y})")

        # リサイズ後の画像上に点を描画
        cv2.circle(resized_image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow(window_name, resized_image)

        # 4点選ばれたら透視変換を実行
        if len(points) == 4:
            perform_perspective_transform()

def perform_perspective_transform():
    """
    透視変換を行う関数
    選択された4点を使用して平面補正する
    """
    global points, original_image
    # 指定された4点
    src_pts = np.array(points, dtype="float32")

    # 変換後の座標（出力画像のサイズを定義）
    width = 3200
    height = 1800
    dst_pts = np.array([
        [0, 0],
        [width-1, 0],
        [width-1, height-1],
        [0, height-1]
    ], dtype="float32")

    # 透視変換行列を計算
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 画像に対して透視変換を実行
    warped = cv2.warpPerspective(original_image, matrix, (width, height))

    # 結果の表示と保存
    cv2.imshow("Warped Image", warped)
    cv2.imwrite(output_path, warped)
    print("Transformation complete. Result saved")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 画像の読み込み
    i= 25
    s =int ((i-1) /4 +1) 
    picture = s*5

    image_path = f"{i}.jpg"
    if i%4 == 1:
        picture_name = "Boat"
    elif i%4 == 2:
        picture_name = "Lenna"
    elif i%4 == 3:
        picture_name = "Music"
    elif i%4 == 0:
        picture_name = "News"
        

    # picture = math.floor((picture+3) / 4) * 5
    output_path = f"{picture_name}{picture}.png"
    # image_path = "0217dng.dng"
    print (output_path)
    
    # path = "40_c"
    # image_path = "m3.jpg"# 任意の画像ファイルに変更
    # output_path = "Cutm3.png"
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Error: Image not found.")
        exit(1)
    
    # 画像を縮小して表示
    resized_image = cv2.resize(original_image, (0, 0), fx=resize_factor, fy=resize_factor)
    
    # ウィンドウ設定
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Please click 4 points on the image in clockwise order.")
    cv2.imshow(window_name, resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
