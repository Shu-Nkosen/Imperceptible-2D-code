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
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_folder, f"frame_{frame_number:05d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_number += 1

    cap.release()
    print(f"{frame_number} 枚のフレームを保存しました。")

# 使用例
video_file_path = "scr.mp4"       # 分割したい動画のファイルパス
output_dir = "frames_output"              # フレームの保存先フォルダ
split_video_into_frames(video_file_path, output_dir)
