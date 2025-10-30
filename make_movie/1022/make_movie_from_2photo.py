import glfw
from OpenGL.GL import *
from PIL import Image
import numpy as np
import os

# ======= ユーザー設定 ==========
# 表示する画像を選択（0: hocho, 1: kosen, 2: nagaoka_fireworks, 3: rice）
SELECTED_IMAGE = 1

# 元画像のベース名リスト
base_image_names = [
    "hocho",
    "kosen",
    "nagaoka_fireworks",
    "rice"
]

# 輝度調整値（ファイル名に使用）
BOTH = 2
BRIGHTNESS_INCREASE = BOTH  # 白部分の輝度増加量
BRIGHTNESS_DECREASE = BOTH  # 黒部分の輝度減少量

# interval framerate / interval
interval = 1

# 各画像の表示回数（カウントベース）[normal, inv]
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

# ファイル名を生成（normal と inv の2つ）
image_files = [
    f"{selected_base_name}_b{BRIGHTNESS_INCREASE}_d{BRIGHTNESS_DECREASE}_normal.png",
    f"{selected_base_name}_b{BRIGHTNESS_INCREASE}_d{BRIGHTNESS_DECREASE}_inv.png"
]

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

# 画像をテクスチャとして読み込み
textures = []
for img_file in image_files:
    img_path = os.path.join(current_dir, img_file)
    if os.path.exists(img_path):
        texture_id = load_texture(img_path)
        textures.append(texture_id)
        print(f"読み込み完了: {img_file}")
    else:
        print(f"Warning: {img_file} が見つかりません")
        glfw.terminate()
        raise FileNotFoundError(f"{img_file} が見つかりません")

# 表示管理用のインデックスとカウンター
current_display_index = 0
frame_display_counter = 0

print(f"\n設定:")
print(f"  選択画像: {selected_base_name} (番号: {SELECTED_IMAGE})")
print(f"  輝度増加量: {BRIGHTNESS_INCREASE}")
print(f"  輝度減少量: {BRIGHTNESS_DECREASE}")
print(f"  リフレッシュ間隔: {interval}")
print(f"  表示パターン: normal ⇔ inv")
print(f"\nプログラム実行中... (ESCキーまたはウィンドウを閉じて終了)")

while not glfw.window_should_close(window):
    # 表示回数が設定値に達したら次へ
    if frame_display_counter >= frame_durations[current_display_index]:
        frame_display_counter = 0
        current_display_index = (current_display_index + 1) % len(textures)
    
    # 現在のテクスチャを描画
    draw_texture(textures[current_display_index])
    
    glfw.swap_buffers(window)
    glfw.poll_events()
    
    frame_display_counter += 1

# 終了処理
glfw.terminate()
print("プログラム終了")