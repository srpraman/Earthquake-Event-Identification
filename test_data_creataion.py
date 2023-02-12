import os
import numpy as np
import random
import shutil

file_list = os.listdir("/home/sairaman/Desktop/stead-dataset/data/reshaped_images/NOISE")
setOfNumbers = set()
numLow = 0
numHigh = 100000
n = 10000
while len(setOfNumbers) < n:
    setOfNumbers.add(random.randint(numLow, numHigh))
#a = np.random.randint(0,100000,(10000))
setOfNumbers = list(setOfNumbers)

for num in setOfNumbers:
    file_name = file_list[num]
    source = "/home/sairaman/Desktop/stead-dataset/data/reshaped_images/NOISE/" + file_name
    dest = "/home/sairaman/Desktop/stead-dataset/data/test_reshaped_images/NOISE/" + file_name
    shutil.move(source,dest)
    
