import cv2
import os

def split_video_into_frames(video_path, output_folder):
    # 動画ファイルを読み込む
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("動画を開けませんでした。")
        return

    # フレームレート（fps）、フレーム数、動画の長さを取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    print(f"動画の長さ: {duration_sec:.8f} 秒")
    print(f"フレーム数: {total_frames}")
    print(f"フレームレート (FPS): {fps}")

    # 出力フォルダを作成
    os.makedirs(output_folder, exist_ok=True)

    frame_number = 0
    max_frames = 120
    while frame_number < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # フレームをフルHD解像度に揃える
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        frame_filename = os.path.join(output_folder, f"frame_{frame_number:05d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_number += 1

    cap.release()
    print(f"{frame_number} 枚のフレームを保存しました。")

# 使用例
prefixes = ["ex", "nagaoka", "hocho", "rice"]
targets = ["B", "G", "I", "R", "X"]
for prefix in prefixes:
    for target in targets:
        for idx in range(1, 10):
            video_file_path = f"{prefix}_{target}{idx}.mp4"
            output_dir = f"{prefix}_{target}{idx}"
            split_video_into_frames(video_file_path, output_dir)
