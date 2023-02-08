
# Importing Image class from PIL module
from PIL import Image
import os

noise_path = "/home/sairaman/Desktop/stead-dataset/data/images/NOISE/"
eq_path = "/home/sairaman/Desktop/stead-dataset/data/images/EQ/"
noise_image_list = os.listdir(noise_path)
eq_image_list = os.listdir(eq_path)
count = 0

'''for image in eq_image_list:
    im = Image.open(eq_path+image)
    width, height = im.size
    
    # Setting the points for cropped image (Cropped image of above dimension)
    left = 80
    top = 58
    right = 570
    bottom = 427

    im1 = im.crop((left, top, right, bottom))
    newsize = (300, 300)
    im1 = im1.resize(newsize)
    im1.save("/home/sairaman/Desktop/stead-dataset/data/reshaped_images/EQ/" + image)
    count = count + 1
    print(f'done for {count}')'''

for image in noise_image_list:
    im = Image.open(noise_path+image)
    width, height = im.size
    
    # Setting the points for cropped image (Cropped image of above dimension)
    left = 80
    top = 58
    right = 570
    bottom = 427

    im1 = im.crop((left, top, right, bottom))
    newsize = (300, 300)
    im1 = im1.resize(newsize)
    im1.save("/home/sairaman/Desktop/stead-dataset/data/reshaped_images/NOISE/" + image)
    count = count + 1
    print(f'done for {count}')