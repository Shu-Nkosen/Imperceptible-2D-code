from PIL import Image
import numpy as np
import random
import reedsolo
import cv2

def reorder_binary_string(binary_string):
    # 最初の64文字をそのまま保持
    first_part = binary_string[:64]
    second_part = ""

    # 文字列の長さを取得
    binary_length = len(binary_string)

    # 残りを16ブロック（64文字ずつ）で処理
    for i in range(16):
        row = 64 + i * 64  # 各ブロックの開始位置
        for j in range(32):  # 各ブロック内で順序を指定して並び替え
            if row + j < binary_length:  # 安全に前半を追加
                second_part += binary_string[row + j]
            if row + j + 32 < binary_length:  # 安全に後半を追加
                second_part += binary_string[row + j + 32]

    # 最初の64文字と並び替えた残りを結合
    result = first_part + second_part
    return result


def clip_image_brightness(input_path, target_amount):
        # 画像を読み込む（グレースケール）
    image = Image.open(input_path).convert('L')

        # 画像データを numpy 配列に変換
    image_array = np.array(image, dtype=np.uint8)

        # 明度をクリッピング
    lower = target_amount
    upper = 255 - target_amount
    clipped_array = np.clip(image_array, lower, upper)

        # numpy 配列を画像に戻す
    clipped_image = Image.fromarray(clipped_array.astype(np.uint8))

    return clipped_image



# バイナリデータにする
def url_to_binary(url):
    """URLをバイナリ文字列に変換"""
    return ''.join(format(ord(char), '08b') for char in url)

def reed_solo_encode(binary_string, ecc_symbols):

    convert_bytes = bytearray(int(binary_string[i:i + 8], 2) for i in range(0, len(binary_string), 8))

    # リードソロモンエンコード
    rs = reedsolo.RSCodec(ecc_symbols)
    encoded = rs.encode(convert_bytes)

    # エンコードされたデータをバイナリ文字列に変換
    encoded_binary = ''.join(format(byte, '08b') for byte in encoded)
    return encoded_binary

def decimal_to_binary(decimal_number2):
    # decimal_number2(文字量)を6文字にクリッピング
    binary_number2 = bin(decimal_number2)[2:].zfill(6)[-6:]
    
    # 結合して32文字に調整
    binary_number = (
        '1' +
        binary_number2 +
        binary_number2 +
        binary_number2 +
        binary_number2 +
        binary_number2 +
        '1'
    )
    return binary_number.zfill(32)[-32:]  # 最終的に32文字に調整

def test_reed_solo(url):
    """
    入力URLをバイナリに変換し、リードソロモン符号を使用してエンコードを行う。
    また、URL長を含むデータを追加して完全なバイナリストリングを生成する。
    
    Args:
        url (str): エンコード対象のURL。
    
    Returns:
        str: URL長データとエンコードされたバイナリデータを結合したバイナリ文字列。
    """
    print(f"Original URL: {url}")

    # URLの長さから必要な誤り訂正シンボル数を計算
    ecc_symbols = 68 - len(url)

    # URLをバイナリ文字列に変換
    binary_string = url_to_binary(url)
    print(f"Binary String Length: {len(binary_string)}")
    print(f"Binary String: {binary_string}")

    # リードソロモン符号でエンコード
    encoded_binary = reed_solo_encode(binary_string, ecc_symbols)
    print(f"Encoded Binary String Length: {len(encoded_binary)}")
    print(f"Encoded Binary String: {encoded_binary}")

    # URL長をバイナリ文字列として追加
    length = len(url)
    binary_number = decimal_to_binary(length)
    print(f"Bit Data String: {binary_number}")
    print("*")

    # URL長データとエンコード済みデータを結合し、最終バイナリ文字列を生成
    print(len(binary_number + encoded_binary))
    print(f"All Binary String: {binary_number}{encoded_binary}")

    return binary_number + encoded_binary

# バイナリデータ終了




# 画像に埋め込む

def embed_binary_data_with_local_brightness(binary_data):
    """
    バイナリデータに基づき、1の場合は明度を2ピクセルごとに上下に変化させる。
    各ピクセルごとに周囲の平均明度を調査して変化量を適応的に決定。
    """
    width=32
    height=18
    
    # クリッピング処理を実行
    img = clip_image_brightness(input_image_path, target_amount)

    # カラー画像の場合、グレースケールに変換
    if img.mode != 'L':
        img = img.convert('L')  # グレースケール変換

    # 画像を1920x1080にリサイズ
    img = img.resize((1920, 1080))
    img_width, img_height = img.size

    # グリッドのサイズを計算
    block_width = img_width // width
    block_height = img_height // height

    for i, bit in enumerate(binary_data):
        row, col = divmod(i, width)
        left, upper = col * block_width, row * block_height
        right, lower = left + block_width, upper + block_height

        # グリッドの領域を取得
        block = img.crop((left, upper, right, lower))
        pixels = np.array(block, dtype=int)  # 一時的にint型に変換

        if bit == '1':  # 1の場合
            for x in range(pixels.shape[0]):
                for y in range(pixels.shape[1]):
                    # 周辺2×2ピクセルの明度を取得
                    x_min = max(0, x - 1)
                    x_max = min(pixels.shape[0], x + 2)
                    y_min = max(0, y - 1)
                    y_max = min(pixels.shape[1], y + 2)

                    local_region = pixels[x_min:x_max, y_min:y_max]
                    local_avg_brightness = np.mean(local_region)



                    # # 周辺の平均明度に基づいて変化量を計算
                    # if local_avg_brightness >= (255 - target_amount):  # ほぼ白
                    #     change_amount = 255 - local_avg_brightness
                    # elif local_avg_brightness <= target_amount:  # ほぼ黒
                    #     change_amount = local_avg_brightness
                    # else:  # 通常
                    change_amount = target_amount

                    Upper = change_amount
                    Lower = change_amount
                    if random.random() < 1/100000:
                        print(Upper, Lower)

                    # 明度を調整
                    if ((x // 2) + (y // 2)) % 2 == 0:
                        # 明度を下げる
                        pixels[x, y] = max(0, pixels[x, y] - Lower )
                    else:
                        # 明度を上げる
                        pixels[x, y] = min(255, pixels[x, y] + Upper)

        # 値をuint8型に変換
        pixels = np.clip(pixels, 0, 255).astype('uint8')

        # 変化を加えたグリッドを画像に貼り付け
        new_block = Image.fromarray(pixels)
        img.paste(new_block, (left, upper))

    # 加工後の画像を保存
    img.save(output_image_path)
    print(f"加工した画像を保存しました: {output_image_path}")


# 動画の設定
frame_size = (1920, 1080)  # 動画の解像度
fps = 10  # フレームレート
repeat_count = 60

# 画像の設定
n = 0.5         # スティーブンスのべき法則のべき指数（明るさの場合）
target_amount = 50 # 輝度変化量
url = "https://www.nagaoka-ct.ac.jp"
binary_data = test_reed_solo (url)
images_to_write = []

for i in range(5, 51, 5): 
    target_amount = i

    binary_data = test_reed_solo(url)

    binary_data = reorder_binary_string(binary_data)

    image_path = "Boat"
    input_image_path = f"{image_path}.png"  # 入力画像ファイルのパス（カラーまたは白黒画像）
    output_image_path = f"{image_path}{target_amount}.png"  # 出力画像ファイルのパス
    embed_binary_data_with_local_brightness(binary_data)

    image_path = "Lena"
    input_image_path = f"{image_path}.png"  # 入力画像ファイルのパス（カラーまたは白黒画像）
    output_image_path = f"{image_path}{target_amount}.png"  # 出力画像ファイルのパス
    embed_binary_data_with_local_brightness(binary_data)

    image_path = "Music"
    input_image_path = f"{image_path}.png"  # 入力画像ファイルのパス（カラーまたは白黒画像）
    output_image_path = f"{image_path}{target_amount}.png"  # 出力画像ファイルのパス
    embed_binary_data_with_local_brightness(binary_data)

    image_path = "News"
    input_image_path = f"{image_path}.png"  # 入力画像ファイルのパス（カラーまたは白黒画像）
    output_image_path = f"{image_path}{target_amount}.png"  # 出力画像ファイルのパス
    embed_binary_data_with_local_brightness(binary_data)

    image_files1 = [f"Boat{target_amount}.png"]
    image_files2 = [f"Lena{target_amount}.png"]
    image_files3 = [f"Music{target_amount}.png"]
    image_files4 = [f"News{target_amount}.png"]

    images_to_write += image_files1 * repeat_count + image_files2 * repeat_count + image_files3 * repeat_count + image_files4 * repeat_count
    # images_to_write +=image_files1 * repeat_count + image_files3 * repeat_count 


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(f'AllAmount.mp4', fourcc, fps, frame_size)

for image_file in images_to_write:
    # 画像を読み込む
    img = cv2.imread(image_file)
    # if img is None:
    #     print(f"Error: Could not load image {image_file}")
    #     continue
    # print(f"Successfully loaded {image_file}")

    # 解像度を動画に合わせる
    img_resized = cv2.resize(img, frame_size)

    # 動画にフレームとして追加
    video_writer.write(img_resized)

    # 動画ライターを解放
video_writer.release()
print(f"Video creation completed: AllAmount.mp4")
print("*")