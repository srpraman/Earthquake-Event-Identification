import os
import numpy as np
import random
import shutil

file_list = os.listdir("/home/sairaman/Desktop/stead-dataset/data/aug_reshaped_images/EQ")
setOfNumbers = set()
numLow = 0
numHigh = 900000
n = 500000
while len(setOfNumbers) < n:
    setOfNumbers.add(random.randint(numLow, numHigh))
#a = np.random.randint(0,100000,(10000))
setOfNumbers = list(setOfNumbers)

for num in setOfNumbers:
    file_name = file_list[num]
    source = "/home/sairaman/Desktop/stead-dataset/data/aug_reshaped_images/EQ/" + file_name
    dest = "/home/sairaman/Desktop/stead-dataset/data/aug_test_reshaped_images/EQ/" + file_name
    shutil.move(source,dest)
    
