import glfw
from OpenGL.GL import *
import time
from collections import deque

# ======= ユーザー設定 ==========
# 輝度値（0〜255）
brightness_values = [0, 50]  # 2つの輝度

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

# タイミング計測用（バッファリング）
last_time = time.perf_counter()
frame_count = 0
switch_times = deque(maxlen=100)  # 最大100回分のログを保持
log_buffer = []

while not glfw.window_should_close(window):
    # 描画処理を最優先
    brightness = brightness_values[current_display_index]
    draw_gray(brightness)
    
    glfw.swap_buffers(window)
    
    # swap後に時間計測（描画に影響しない）
    current_time = time.perf_counter()
    
    glfw.poll_events()

    # カウンターを増やす
    frame_display_counter += 1
    frame_count += 1

    # 表示回数が設定値に達したら次へ切り替え
    if frame_display_counter >= frame_durations[current_display_index]:
        if len(switch_times) > 0:
            switch_duration = current_time - switch_times[-1]
            # printせずバッファに保存
            log_buffer.append(f"Brightness {brightness}: {switch_duration*1000:.2f}ms ({frame_display_counter}f)")
        
        switch_times.append(current_time)
        frame_display_counter = 0
        current_display_index = (current_display_index + 1) % len(brightness_values)

    # 1秒ごとにまとめて表示（描画ループへの影響を最小化）
    # if current_time - last_time >= 1.0:
    #     fps = frame_count / (current_time - last_time)
        
    #     # バッファの内容を一括出力
    #     if log_buffer:
    #         print("\n".join(log_buffer))
    #         log_buffer.clear()
        
    #     print(f"=== FPS: {fps:.1f} Hz ===\n")
        
    #     frame_count = 0
    #     last_time = current_time

# 終了処理
glfw.terminate()