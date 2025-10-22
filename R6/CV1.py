import cv2

def process_image(input_file, grayscale_output, edge_output, low_threshold=50, high_threshold=150):
    """
    指定した画像をグレースケール変換し、エッジ検出を行うプログラム

    Parameters:
        input_file (str): 入力画像のファイル名
        grayscale_output (str): グレースケール画像の保存先ファイル名
        edge_output (str): エッジ検出後の画像の保存先ファイル名
        low_threshold (int): Cannyエッジ検出の低しきい値（デフォルト: 50）
        high_threshold (int): Cannyエッジ検出の高しきい値（デフォルト: 150）
    """

    # 画像を読み込み
    image = cv2.imread(input_file)
    if image is None:
        print(f"エラー: 画像 '{input_file}' を読み込めません。")
        return
    
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Canny エッジ検出
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    # 変換後の画像を保存
    cv2.imwrite(grayscale_output, gray)
    cv2.imwrite(edge_output, edges)
    
    print(f"✅ グレースケール画像を '{grayscale_output}' に保存しました。")
    print(f"✅ エッジ検出画像を '{edge_output}' に保存しました。")

# === ここから実行部分 ===
if __name__ == "__main__":
    # 入力画像と出力画像のファイル名を指定
    file = "16"  # 変換したい画像のファイル名
    input_file = f"{file}.jpg"  # 変換したい画像のファイル名
    grayscale_output = f"{file}output_grayscale.jpg"  # グレースケール画像の保存先
    edge_output = f"{file}output_edges.jpg"  # エッジ検出画像の保存先

    # 画像処理を実行
    process_image(input_file, grayscale_output, edge_output)
