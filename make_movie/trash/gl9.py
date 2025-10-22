import glfw
from OpenGL.GL import *
from PIL import Image
import time
import glob

# 初期化
glfw.init()
window = glfw.create_window(800, 600, "Player", None, None)
glfw.make_context_current(window)
glEnable(GL_TEXTURE_2D)

# テクスチャ読み込み
frames = []
for path in sorted(glob.glob("frames/frame*.png")):
    img = Image.open(path).convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, img.tobytes())
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    frames.append(tex)

# 再生ループ
fps = 60
dt = 1 / fps
repeat = 100
for _ in range(repeat):
    for tex in frames:
        if glfw.window_should_close(window): break
        t0 = time.time()
        glClear(GL_COLOR_BUFFER_BIT)
        glBindTexture(GL_TEXTURE_2D, tex)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(-1, -1)
        glTexCoord2f(1, 0); glVertex2f( 1, -1)
        glTexCoord2f(1, 1); glVertex2f( 1,  1)
        glTexCoord2f(0, 1); glVertex2f(-1,  1)
        glEnd()
        glfw.swap_buffers(window)
        glfw.poll_events()
        time.sleep(max(0, dt - (time.time() - t0)))

glfw.terminate()
