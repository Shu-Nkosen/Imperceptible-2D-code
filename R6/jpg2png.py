from PIL import Image

fileName = "mi4"

imagefile = Image.open(rf"{fileName}.jpg")
imagefile.save(rf"{fileName}.png")