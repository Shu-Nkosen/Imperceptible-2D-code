from PIL import Image

def convertToGray(inputImage, outputImage):
    image = Image.open(inputImage)
    gray = image.convert('L')
    gray.save(outputImage)

inputImage = "News.png"
outputImage = "News_gray.png"

convertToGray(inputImage, outputImage)
