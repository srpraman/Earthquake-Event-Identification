import os
import numpy as np
import random
import shutil

file_list = os.listdir("/home/raman/Desktop/stead_dataset/data/chunk2_ir_removed")
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
    source = "/home/raman/Desktop/stead_dataset/data/chunk2_ir_removed/" + file_name
    dest = "/home/raman/Desktop/stead_dataset/data/test/eq/" + file_name
    shutil.copy(source,dest)
    
