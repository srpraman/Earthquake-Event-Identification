import numpy as np 
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

file_list=os.listdir("/home/raman/Desktop/stead_dataset/data/chunk2_ir_removed")
df=pd.read_csv("/home/raman/Desktop/stead_dataset/csv_files/updated_chunk2.csv")
def magnitude(file_name,df):
    sl_no = int(file_name.split("_")[-1].split(".")[0])
    #return sl_no
    return df.iloc[sl_no]["source_magnitude"]
mag = []
for file in file_list:
    try:
        mag.append((magnitude(file,df)))
    except:
        print(f"mag not available for {file}")
        pass
sns.histplot(data=mag,bins=6)
plt.savefig("a.png")