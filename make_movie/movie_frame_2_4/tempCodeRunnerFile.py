
        return
    
    # 画像のサイズを取得
    img_h, img_w = image.shape[:2]
    
    # 正方形領域のサイズを計算（高さに合わせる）
    square_size = img_h
    
    # 画像の中央に正方形領域を配置するためのオフセット
    x_offset = (img_w - square_size) // 2
    
    # QRコードを正方形サイズにリサイズ
    qr_resized = cv2.resize(qr_code, (square_size, square_size))
    