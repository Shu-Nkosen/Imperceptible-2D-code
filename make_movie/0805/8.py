import glfw
from OpenGL.GL import *
import time

# ======= ユーザー設定 ==========
# 輝度値（0〜255）
brightness_values = [0, 255]  # 2つの輝度

#interval framerate / interval
interval = 1

# フレーム表示回数（カウントベース）
frame_durations = [1, 1]
# ==============================

def draw_gray(brightness):
    norm = brightness / 255.0  # 正規化（0.0〜1.0）
    glClearColor(norm, norm, norm, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)

# GLFW 初期化
if not glfw.init():
    raise Exception("GLFW initialization failed")

window = glfw.create_window(800, 600, "Gray Flicker", None, None)
glfw.make_context_current(window)

# 垂直同期ON（1フレーム = モニターの1リフレッシュ周期）
glfw.swap_interval(interval)

# 表示管理用のインデックスとカウンター
current_display_index = 0
frame_display_counter = 0

while not glfw.window_should_close(window):
    # 表示回数が設定値に達したら次へ
    if frame_display_counter >= frame_durations[current_display_index]:
        frame_display_counter = 0
        current_display_index = (current_display_index + 1) % len(frame_durations)

    # 輝度に応じてグレーで描画
    brightness = brightness_values[current_display_index]
    draw_gray(brightness)

    glfw.swap_buffers(window)
    glfw.poll_events()

    frame_display_counter += 1

# 終了処理
glfw.terminate()
