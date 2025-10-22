import numpy as np
from PIL import Image
import reedsolo

def restore_original_order(reordered_binary_string):
    # 最初の64文字をそのまま保持
    first_part = reordered_binary_string[:64]
    second_part = reordered_binary_string[64:]

    # 空のリストを用意して元の並びを再構築
    original_second_part = [''] * len(second_part)

    # 残りを16ブロック（64文字ずつ）で元に戻す
    for i in range(16):
        row = i * 64  # ブロックの開始位置
        for j in range(32):
            if row + 2 * j < len(second_part):
                # 元の並びの前半部分を配置
                original_second_part[row + j] = second_part[row + 2 * j]
            if row + 2 * j + 1 < len(second_part):
                # 元の並びの後半部分を配置
                original_second_part[row + j + 32] = second_part[row + 2 * j + 1]

    # 元の文字列を再構築
    original_binary_string = first_part + ''.join(original_second_part)
    return original_binary_string

def reed_solo_decode(encoded_binary):
    """
    URLの長さ情報で3回以上一致する値を採用してURLの長さを決定する。
    そこから誤り訂正符号数を計算
    
    Args:
        encoded_binary (str): エンコードされたバイナリ文字列。

    Returns:
        str: デコードされたバイナリ文字列。
    """

    # URL長データリスト
    length_candidates = [
        int(encoded_binary[1:7], 2),
        int(encoded_binary[7:13], 2),
        int(encoded_binary[13:19], 2),
        int(encoded_binary[19:25], 2),
        int(encoded_binary[25:31], 2)
    ]

    print (length_candidates)


    from collections import Counter
    length_counts = Counter(length_candidates)

    valid_lengths = [length for length, count in length_counts.items() if count >= 3]

    if not valid_lengths:
        print("データ数読み込みエラーです")
        return None
    else:
        url_length = max(valid_lengths)
        url_binary = encoded_binary[32:]
        ecc_symbols = 68 - url_length

        encoded_bytes = bytearray(int(url_binary[i:i + 8], 2) for i in range(0, len(url_binary), 8))

        print (encoded_bytes)

        # リードソロモンデコード
        try:
            rs = reedsolo.RSCodec(ecc_symbols)
            decoded = rs.decode(encoded_bytes)
            # デコード結果がタプルの場合、最初の要素を取得
            if isinstance(decoded, tuple):
                decoded = decoded[0]

            # デコード結果をバイナリ文字列に変換
            decoded_binary = ''.join(format(byte, '08b') for byte in decoded)
            return decoded_binary
        except reedsolo.ReedSolomonError:
            print("デコードエラーです")
            return None


def binary_to_url(binary_string):
    """バイナリ文字列をURLに変換"""
    return ''.join(chr(int(binary_string[i:i + 8], 2)) for i in range(0, len(binary_string), 8))

def decode_brightness(image_path, threshold, score_threshold):
    """
    画像を解析し、タイルの中心24×24領域を考慮してバイナリ文字列を生成する。

    Parameters:
        image_path (str): 入力画像のファイルパス
        threshold (int): 明度差の閾値（デフォルト: ±50）
        score_threshold (int): 規則性スコアの判定閾値（デフォルト: 70%）

    Returns:
        str: 32×18タイルの解析結果を示すバイナリ文字列
    """
    # 入力画像を読み込み
    image = Image.open(image_path).convert('L')
    width, height = image.size

    # タイルのサイズ
    tile_width, tile_height = width // 32, height // 18
    binary_result = []

    # 32×18のタイルを解析
    for tile_row in range(18):
        for tile_col in range(32):
            # タイルを切り出す
            left = tile_col * tile_width
            upper = tile_row * tile_height
            right = left + tile_width
            lower = upper + tile_height
            tile = image.crop((left, upper, right, lower))

            # 外側3格子分をカットして中心領域(24x24)を解析
            # center_tile = crop_to_center(tile, 3)
            center_tile = tile

            # 各タイルの解析結果を取得
            is_encoded = analyze_tile(center_tile, threshold, score_threshold)
            binary_result.append('1' if is_encoded else '0')

    # 結果を文字列として返す
    return ''.join(binary_result)


def crop_to_center(tile, margin):
    """
    タイルの外側をカットし、中心領域を取得。

    Parameters:
        tile (PIL.Image): 解析対象のタイル画像
        margin (int): 外側の格子分をカットするピクセル数

    Returns:
        PIL.Image: 中心領域を抽出したタイル
    """
    tile_width, tile_height = tile.size
    left = margin
    upper = margin
    right = tile_width - margin
    lower = tile_height - margin
    return tile.crop((left, upper, right, lower))


def analyze_tile(tile, threshold, score_threshold):
    """
    1タイルを解析し、格子状パターンの規則性を評価。

    Parameters:
        tile (PIL.Image): 解析対象のタイル画像
        threshold (int): 明度差の閾値
        score_threshold (int): 規則性スコアの判定閾値

    Returns:
        bool: エンコードがある（True）かない（False）か
    """
    # タイル内を30*30格子状に分割
    grid_rows, grid_cols = 30, 30
    tile_array = np.array(tile)

    # 格子状パターンの規則性を評価
    score = evaluate_pattern(tile_array, grid_rows, grid_cols , threshold)

    # 規則性スコアが閾値以上であればエンコードあり
    return score >= score_threshold


def evaluate_pattern(tile_array, grid_rows, grid_cols, threshold):
    """
    格子状パターンの一方向変化の規則性を評価。
    3つの中心位置 (0.17, 0.5, 0.83) のうち、規則性が最大となるものを選択。

    Parameters:
        tile_array (np.array): タイルの明度配列
        grid_rows (int): 格子の行数
        grid_cols (int): 格子の列数
        threshold (float): 明度差の閾値（使われていないが保持）

    Returns:
        float: 規則性スコア（0～100%）
    """
    tile_height, tile_width = tile_array.shape
    cell_height = tile_height / grid_rows
    cell_width = tile_width / grid_cols

    # 中心位置候補
    center_factors = [0.17, 0.5, 0.83]
    best_consistent_cells = 0
    best_total_cells = 0

    for factor in center_factors:
        consistent_cells = 0
        total_cells = 0

        # 各格子の中心ピクセルの明度を取得
        for row in range(grid_rows):
            for col in range(grid_cols):
                center_y = int((row + factor) * cell_height)
                center_x = int((col + factor) * cell_width)

                # 現在の格子の中心明度
                if center_y < tile_height and center_x < tile_width:
                    pixel_value = tile_array[center_y, center_x]
                    plus = 0
                    minus = 0

                    # 四方の隣接格子の明度と比較
                    if col + 1 < grid_cols:  # 右隣
                        neighbor_x = int((col + 1 + factor) * cell_width)
                        neighbor_value = tile_array[center_y, neighbor_x]
                        if pixel_value < neighbor_value:
                            plus += 1
                        elif pixel_value > neighbor_value:
                            minus += 1
                    if col - 1 >= 0:  # 左隣
                        neighbor_x = int((col - 1 + factor) * cell_width)
                        neighbor_value = tile_array[center_y, neighbor_x]
                        if pixel_value < neighbor_value:
                            plus += 1
                        elif pixel_value > neighbor_value:
                            minus += 1
                    if row + 1 < grid_rows:  # 下隣
                        neighbor_y = int((row + 1 + factor) * cell_height)
                        neighbor_value = tile_array[neighbor_y, center_x]
                        if pixel_value < neighbor_value:
                            plus += 1
                        elif pixel_value > neighbor_value:
                            minus += 1
                    if row - 1 >= 0:  # 上隣
                        neighbor_y = int((row - 1 + factor) * cell_height)
                        neighbor_value = tile_array[neighbor_y, center_x]
                        if pixel_value < neighbor_value:
                            plus += 1
                        elif pixel_value > neighbor_value:
                            minus += 1

                    # 規則的（すべて同じ方向に変化している）場合
                    if plus == 4 or minus == 4:
                        consistent_cells += 1
                    total_cells += 1

        # 現在の中心位置の結果が最良か確認
        if consistent_cells > best_consistent_cells:
            best_consistent_cells = consistent_cells
            best_total_cells = total_cells

    # スコアを計算
    if best_total_cells == 0:
        return 0  # 評価可能なセルがない場合はスコア0
    return best_consistent_cells / best_total_cells * 100

set_data = "101110001110001110001110001110010110100001110100011101000111000001110011001110100010111100101111011101110111011101110111001011100110111001100001011001110110000101101111011010110110000100101101011000110111010000101110011000010110001100101110011010100111000010101110000111000110010100101111110010100011111101100001001011110010110000001100001000100110001111000000001101110011011100000110111001000010100110010011111110100000110001110100100111001001000100000001110101000111010001000111010001101101000111000010010000101001010000011110001001001000011110100111000110011100000011010101"

score_threshold = 50
# brightness = 50
position = "C"
# image_path = f"Lenna{brightness}{position}.png"  # 入力画像ファイル名
# image_path = f"News{brightness}.png"  # 入力画像ファイル名

for brightness in range(5, 51, 5):
    image_path = f"Lenna{brightness}.png"
    # image_path = f"News{brightness}.png"
    # image_path = f"Music{brightness}.png"
    image_path = f"Boat{brightness}.png"

    # 実行部分
    if __name__ == "__main__":
        result = decode_brightness(image_path, brightness*0.8, score_threshold)
        # print("画像のデータ:", result)
        decode_data = restore_original_order(result)
        print("並び替えたバイナリデータ:", decode_data)
        # 異なる文字をカウント
        differences = sum(1 for s, d in zip(set_data, decode_data) if s != d)
        print(f"異なる文字の数: {differences}")
        
        decode_reedSolo = reed_solo_decode(decode_data)
        print(image_path)
        if decode_reedSolo:
            url = binary_to_url(decode_reedSolo)
            print("デコードされたURL:", url)
        print("*")
