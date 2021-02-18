from PIL import Image
import os
from os.path import join
from matplotlib import pyplot as plt


# Function to change the image size
def changeImageSize(maxWidth,
                    maxHeight,
                    image):
    widthRatio = maxWidth / image.size[0]
    heightRatio = maxHeight / image.size[1]

    newWidth = int(widthRatio * image.size[0])
    newHeight = int(heightRatio * image.size[1])

    newImage = image.resize((newWidth, newHeight))
    return newImage


# Take two images for blending them together
path = '../../data/geoPose3K/eth_ch1_2011-10-04_14_25_54_01024'
image1 = Image.open(join(path, "photo.jpg"))
image2 = Image.open(join(path, "depth.png"))

# Make the images of uniform size
image3 = changeImageSize(1500, 1000, image1)
image4 = changeImageSize(1500, 1000, image2)

# Make sure images got an alpha channel
image5 = image3.convert("RGBA")
image6 = image4.convert("RGBA")

# Display the images
# image6.show()
# image5.show()


# alpha-blend the images with varying values of alpha
alphaBlended1 = Image.blend(image5, image6, alpha=.2)
alphaBlended2 = Image.blend(image5, image6, alpha=.4)

# Display the alpha-blended images
# alphaBlended1.show()
alphaBlended2.save(join(path, 'blend.png'))
# plt.imshow(alphaBlended2)
# # plt.show()
#
# plt.imsave(join(path, 'blend.png'), alphaBlended2)
