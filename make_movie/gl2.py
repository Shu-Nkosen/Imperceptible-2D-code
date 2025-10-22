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
frame_paths = sorted(glob.glob("frames/Music*.png"))
textures = [load_texture(path) for path in frame_paths]

# フレームの表示期間を定義
# 例1: frame0を9回、frame1を1回表示 (つまり0*9, 1*1)
frame_durations = [29, 1]

# 例2: frame0を59回、frame1を1回表示 (つまり0*59, 1*1)
# frame_durations = [59, 1]

# frame_durationsの要素数がtexturesの要素数と一致しない場合は調整が必要です
# 今の例だとtexturesにはframe0とframe1の2枚が入っている想定です。

current_display_frame_index = 0  # 現在表示すべきtexturesリストのインデックス
frame_display_counter = 0        # 現在のフレームを表示している回数

# 繰り返し回数は不要になります。フレームの表示制御はループ内で行います。

while not glfw.window_should_close(window):
    # フレームの切り替え判定
    # 現在のフレームが規定の表示回数に達したか？
    if frame_display_counter >= frame_durations[current_display_frame_index]:
        frame_display_counter = 0  # カウンターをリセット
        # 次のフレームインデックスへ進む。リストの最後まで行ったら先頭に戻る。
        current_display_frame_index = (current_display_frame_index + 1) % len(frame_durations)

    glClear(GL_COLOR_BUFFER_BIT)
    
    # current_display_frame_index で示されるフレームを表示
    tex_id, _, _ = textures[current_display_frame_index]
    glBindTexture(GL_TEXTURE_2D, tex_id)
    draw_quad()

    glfw.swap_buffers(window)
    glfw.poll_events()
    glfw.swap_interval(1)  # 180Hz / 3 = 60fps (つまり1フレームあたり約0.0166秒)

    frame_display_counter += 1 # カウンターを増やす

# --- 終了処理 ---
glfw.terminate()