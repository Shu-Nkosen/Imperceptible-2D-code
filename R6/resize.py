from PIL import Image
import os

def resize_and_convert_to_grayscale(input_path, output_path, width, height):
    """
    画像をグレースケールに変換し、解像度を変更する関数

    :param input_path: 入力画像ファイルのパス
    :param output_path: 出力画像ファイルのパス
    :param width: 新しい幅（整数型）
    :param height: 新しい高さ（整数型）
    """
    try:
        # 画像を開く
        with Image.open(input_path) as img:
            # グレースケールに変換
            grayscale_img = img.convert('L')
            # 画像をリサイズ
            resized_img = grayscale_img.resize((width, height), Image.Resampling.LANCZOS)
            # PNG形式で保存
            resized_img.save(output_path, format="PNG")
        print(f"画像が正常にグレースケール化およびリサイズされ、保存されました: {output_path}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    # 入力ファイルパスと出力ファイルパス
    image_path = "Boat"
    # 新しい解像度
    j = 1
    new_width = int(1920 / j)  # 幅を整数型に変換
    new_height = int(1080 / j)  # 高さを整数型に変換

    # 複数画像を処理
    for i in range(5, 51, 5): 
        input_file = f"{image_path}{i}.png"  # 入力画像ファイル名
        output_file = f"{image_path}{i}resize.png"  # 出力画像ファイル名
    
        # 関数を呼び出し
        resize_and_convert_to_grayscale(input_file, output_file, new_width, new_height)
        
    # 単一画像を処理
    input_file = f"{image_path}.png"  # 入力画像ファイル名
    output_file = f"{image_path}resize.png"  # 出力画像ファイル名
    resize_and_convert_to_grayscale(input_file, output_file, new_width, new_height)
