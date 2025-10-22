import matplotlib.pyplot as plt

# データ入力
brightness_values = [
    0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100,
    110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210,
    220, 230, 235, 240, 245, 250, 255
]
luminance_values = [
    5.47, 5.64, 6.20, 6.43, 6.45, 6.77, 7.12, 8.52, 10.4, 12.7,
    15.6, 19.4, 24.1, 29.2, 35.3, 41.9, 49.5, 57.5, 66.1, 75.8,
    85.3, 96, 107, 119, 131, 145, 159, 167, 174, 182, 190, 197
]

# グラフ作成
plt.figure(figsize=(10, 6))
plt.plot(brightness_values, luminance_values, marker='o')
plt.title('Brightness vs Luminance')
plt.xlabel('setting brightness 0-255')
plt.ylabel('measured value cd/m²')
plt.grid(True)
plt.tight_layout()
plt.show()