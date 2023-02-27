import os
import matplotlib.pyplot as plt
import obspy
import numpy as np

x = np.linspace(0,180,num=4500)
file_list = os.listdir("/home/sairaman/Desktop/stead-dataset/data/waveforms/chunk2_ir_removed")
data = []
label = []
os.chdir("/home/sairaman/Desktop/stead-dataset/data/waveforms/chunk2_ir_removed")
for file_name in file_list:
    print(file_name)
    st = obspy.read(file_name)
    st_copy = st.copy()
    tr = st_copy[0]
    tr.interpolate(sampling_rate=25.0)
    tr.filter('bandpass', freqmin=2, freqmax=10, corners=2, zerophase=True)
    if len(tr.data) == 4500:
        data.append(tr.data[1000: ]/max(tr.data[1000: ]))
        #tr.plot()
        #plt.plot(x,tr.data)
        #plt.show()
        label.append(file_name)
    elif len(tr.data) == 4501:
        data.append(tr.data[1000:-1]/max(tr.data[1000:-1]))
        label.append(file_name)
    else:
        pass
        
np.savez_compressed('/home/sairaman/Desktop/stead-dataset/data/waveforms/chunk2_to_1d_array_normalized.npz', **dict(zip(label,data)))
# print(np.load('/home/sairaman/Desktop/stead-dataset/data/waveforms/chunk2_to_1d_array.npz').files)