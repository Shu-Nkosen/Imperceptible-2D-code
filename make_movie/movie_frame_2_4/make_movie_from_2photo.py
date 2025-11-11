import glfw
from OpenGL.GL import *
from PIL import Image
import numpy as np
import os
import gc
import time
from collections import deque
import ctypes

# Windowsの時間分解能を1msに設定（重要！）
try:
    winmm = ctypes.WinDLL('winmm')
    winmm.timeBeginPeriod(1)
    print("Windows timer resolution set to 1ms")
except:
    print("Warning: Could not set timer resolution")

# ======= ユーザー設定 ==========
# 表示する画像を選択（0: hocho, 1: kosen, 2: nagaoka_fireworks, 3: rice）
SELECTED_IMAGE = 4

# 元画像のベース名リスト
base_image_names = [
    "hocho",
    "kosen",
    "nagaoka_fireworks",
    "rice",
    "ex"
]

# 輝度調整値（ファイル名に使用）
BOTH = 2
BRIGHTNESS_INCREASE = BOTH  # 白部分の輝度増加量
BRIGHTNESS_DECREASE = BOTH  # 黒部分の輝度減少量

# interval framerate / interval
interval = 1  # 180Hzで動作

# 各画像の表示回数（カウントベース）
# 1の場合: normal/invを交互に表示
# それ以外（例: 40）: 元画像を表示
frame_durations = [1, 1]
# ==============================

def load_texture(image_path):
    """画像をOpenGLテクスチャとして読み込む"""
    img = Image.open(image_path)
    
    # 1920x1080にリサイズ
    img = img.resize((1920, 1080), Image.LANCZOS)
    
    img_data = np.array(img, dtype=np.uint8)
    
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    
    # テクスチャパラメータ設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    
    # テクスチャデータを転送
    if img.mode == "RGB":
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    elif img.mode == "RGBA":
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
    
    # メモリを明示的に解放
    del img_data
    del img
    
    return texture_id

def draw_texture(texture_id):
    """テクスチャを画面全体に描画"""
    glClear(GL_COLOR_BUFFER_BIT)
    
    # 正投影行列を設定
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1, 1, -1, 1, -1, 1)
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1); glVertex2f(-1, -1)
    glTexCoord2f(1, 1); glVertex2f(1, -1)
    glTexCoord2f(1, 0); glVertex2f(1, 1)
    glTexCoord2f(0, 0); glVertex2f(-1, 1)
    glEnd()
    
    glDisable(GL_TEXTURE_2D)

# カレントディレクトリを取得
current_dir = os.path.dirname(os.path.abspath(__file__))

# 選択された画像のベース名を取得
selected_base_name = base_image_names[SELECTED_IMAGE]

# GLFW 初期化
if not glfw.init():
    raise Exception("GLFW initialization failed")

window = glfw.create_window(1920, 1080, "Image Flicker", None, None)

if not window:
    glfw.terminate()
    raise Exception("Window creation failed")

glfw.make_context_current(window)

# 垂直同期ON
glfw.swap_interval(interval)

# ===== 最適化: 必要な画像のみを一度だけ読み込み =====
# テクスチャキャッシュ（ファイル名 → texture_id）
texture_cache = {}

# 必要な3種類の画像ファイル名
normal_file = f"{selected_base_name}_b{BRIGHTNESS_INCREASE}_d{BRIGHTNESS_DECREASE}_normal.png"
inv_file = f"{selected_base_name}_b{BRIGHTNESS_INCREASE}_d{BRIGHTNESS_DECREASE}_inv.png"
orig_file = f"{selected_base_name}.png"

# 各画像を1回だけ読み込み
for img_file in [normal_file, inv_file, orig_file]:
    img_path = os.path.join(current_dir, img_file)
    if os.path.exists(img_path):
        texture_id = load_texture(img_path)
        texture_cache[img_file] = texture_id
        print(f"読み込み完了: {img_file} (ID: {texture_id})")
    else:
        print(f"Warning: {img_file} が見つかりません")

# frame_durationsに基づいて表示するテクスチャIDのリストを生成
texture_sequence = []
is_normal = True  # normal/invを交互に切り替えるフラグ

for duration in frame_durations:
    if duration == 1:
        # 1の場合: normal/invを交互に
        if is_normal:
            texture_sequence.append(texture_cache[normal_file])
        else:
            texture_sequence.append(texture_cache[inv_file])
        is_normal = not is_normal
    else:
        # 1以外の場合: 元画像を表示
        texture_sequence.append(texture_cache[orig_file])

# テクスチャ読み込み後に明示的にガベージコレクション
gc.collect()

print(f"\n表示パターン:")
is_normal = True
for i, duration in enumerate(frame_durations):
    if duration == 1:
        img_type = "normal" if is_normal else "inv"
        is_normal = not is_normal
    else:
        img_type = "original"
    print(f"  {i}: {img_type} ({duration}フレーム)")

# 表示管理用のインデックスとカウンター
current_display_index = 0
frame_display_counter = 0

# デバッグ用
frame_times = deque(maxlen=100)
swap_times = deque(maxlen=1000)
last_switch_time = time.perf_counter()

print(f"\n設定:")
print(f"  選択画像: {selected_base_name} (番号: {SELECTED_IMAGE})")
print(f"  輝度増加量: {BRIGHTNESS_INCREASE}")
print(f"  輝度減少量: {BRIGHTNESS_DECREASE}")
print(f"  リフレッシュ間隔: {interval}")
print(f"  ユニークなテクスチャ数: {len(texture_cache)}")
print(f"  表示シーケンス長: {len(texture_sequence)}")
print(f"\nプログラム実行中... (ESCキーまたはウィンドウを閉じて終了)")

total_frames = 0
last_frame_time = time.perf_counter()

while not glfw.window_should_close(window):
    # 現在のテクスチャを描画（テクスチャIDを直接使用）
    draw_texture(texture_sequence[current_display_index])
    
    # swap直前の時刻
    pre_swap_time = time.perf_counter()
    glfw.swap_buffers(window)
    post_swap_time = time.perf_counter()
    
    # swapにかかった時間を記録
    swap_duration = (post_swap_time - pre_swap_time) * 1000
    swap_times.append(swap_duration)
    
    glfw.poll_events()
    
    # カウンターを増やす
    frame_display_counter += 1
    total_frames += 1

    # 表示回数が設定値に達したら次へ切り替え
    if frame_display_counter >= frame_durations[current_display_index]:
        switch_duration = (post_swap_time - last_switch_time) * 1000
        frame_times.append(switch_duration)
        
        # 切り替えごとに表示
        expected = frame_durations[current_display_index] * (1000.0 / 180.0)
        print(f"Image {current_display_index}: {switch_duration:.2f}ms (expected={expected:.2f}ms, frames={frame_display_counter})")
        
        frame_display_counter = 0
        current_display_index = (current_display_index + 1) % len(texture_sequence)
        last_switch_time = post_swap_time

# 終了処理
for texture_id in texture_cache.values():
    glDeleteTextures([texture_id])

# タイマー分解能を戻す
try:
    winmm.timeEndPeriod(1)
except:
    pass

glfw.terminate()
print("プログラム終了")