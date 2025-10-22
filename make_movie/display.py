import pyglet

window = pyglet.window.Window()

fps_display = pyglet.window.FPSDisplay(window)  # ← これが今の書き方

@window.event
def on_draw():
    window.clear()
    fps_display.draw()

def update(dt):
    pass  # 今回は特に何も更新しないけど、動かしたい処理があればここに書く

pyglet.clock.schedule_interval(update, 1/60.0)  # 60fpsで update を呼ぶ

pyglet.app.run()
