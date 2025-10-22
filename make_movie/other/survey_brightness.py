import glfw
from OpenGL.GL import *
import time

brightness = 255  # 表示したい輝度（0〜255）

if not glfw.init():
    raise Exception("GLFW initialization failed")

window = glfw.create_window(1600, 800, "Gray Display", None, None)
glfw.make_context_current(window)

# 垂直同期は不要なので削除またはコメントアウト
# glfw.swap_interval(1)

# 一度だけ描画
norm = brightness / 255.0
glClearColor(norm, norm, norm, 1.0)
glClear(GL_COLOR_BUFFER_BIT)
glfw.swap_buffers(window)

# 60秒間そのまま表示
start_time = time.time()
while not glfw.window_should_close(window):
    if time.time() - start_time >= 10:
        break
    glfw.poll_events()  # 終了操作を検知するためだけに実行
    time.sleep(0.1)  # CPU負荷を減らすためにスリープを入れるaaaaaaaaaaaaaaaaaaaaa

glfw.terminate()
