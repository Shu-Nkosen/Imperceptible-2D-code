import qrcode

def generate_qr_code(url, filename, box_size=10, border=4):
    """
    URLを含むQRコードを生成して保存する関数。

    Parameters:
        url (str): QRコードに埋め込むURL。
        filename (str): 保存するファイル名（デフォルトは 'qrcode.png'）。
        box_size (int): QRコードの各ボックスのサイズ（ピクセル単位）。
        border (int): QRコードの余白（ボックス数単位）。
    """
    try:
        # QRコードの設定
        qr = qrcode.QRCode(
            version=1,  # サイズ（1～40）。1は最小。
            error_correction=qrcode.constants.ERROR_CORRECT_L,  # エラー修正レベル
            box_size=box_size,  # 各ボックスのサイズ
            border=border,  # 余白
        )
        # URLを追加
        qr.add_data(url)
        qr.make(fit=True)

        # QRコードを画像として生成
        img = qr.make_image(fill_color="black", back_color="white")

        # ファイルに保存
        img.save(filename)
        print(f"QRコードを {filename} に保存しました。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

# 使用例
generate_qr_code("https://www.nagaoka-ct.ac.jp", "HP_QR.png")
