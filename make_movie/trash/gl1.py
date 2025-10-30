import glfw
from OpenGL.GL import *
from PIL import Image
import time
import glob

def load_texture(image_path):
    img = Image.open(image_path).convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
    img_data = img.tobytes()
    width, height = img.size

    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, img_data)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    return tex_id, width, height


def draw_quad():
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(-1, -1)
    glTexCoord2f(1, 0); glVertex2f(1, -1)
    glTexCoord2f(1, 1); glVertex2f(1, 1)
    glTexCoord2f(0, 1); glVertex2f(-1, 1)
    glEnd()

# GLFW 初期化
if not glfw.init():
    raise Exception("GLFW initialization failed")

window = glfw.create_window(800, 600, "Image Player", None, None)
glfw.make_context_current(window)
glEnable(GL_TEXTURE_2D)


# --- 画像読み込み ---
frame_paths = sorted(glob.glob("frames/frame*.png"))
textures = [load_texture(path) for path in frame_paths]

# window_duration = 1.0/180
# frame_duration = window_duration*2 + 0.001
frame_index = 0

repeat_count = 100
for _ in range(repeat_count):
    frame_index = 0
    while not glfw.window_should_close(window) and frame_index < len(textures):
        # start_time = time.time()

        glClear(GL_COLOR_BUFFER_BIT)
        
        tex_id, _, _ = textures[frame_index]
        glBindTexture(GL_TEXTURE_2D, tex_id)
        draw_quad()

        glfw.swap_buffers(window)
        glfw.poll_events()
        # glFinish() #swap_buffers()の後に読んでも描画切り替えを待ってくれない　垂直同期とは別っぽい
        glfw.swap_interval(3)  # 180Hz / 3 = 60fps

        frame_index += 1
        # elapsed = time.time() - start_time
        # time.sleep(max(0, frame_duration))
        

# --- 終了処理 ---
glfw.terminate()
