import os
import matplotlib.pyplot as plt
import obspy
import numpy as np

x = np.linspace(0,180,num=4500)
file_list = os.listdir("/home/sairaman/Desktop/stead-dataset/data/waveforms/chunk1_ir_removed")
data = []
label = []
file_list = file_list[0:15000]
count = 0
os.chdir("/home/sairaman/Desktop/stead-dataset/data/waveforms/chunk1_ir_removed")
for file_name in file_list:
    print(file_name)
    st = obspy.read(file_name)
    st_copy = st.copy()
    tr = st_copy[0]
    tr.interpolate(sampling_rate=25.0)
    tr.filter('bandpass', freqmin=2, freqmax=10, corners=2, zerophase=True)
    if len(tr.data) == 4500:
        data.append(tr.data/max(tr.data))
        count = count + 1
        #tr.plot()
        #plt.plot(x,tr.data)
        #plt.show()
        label.append(file_name)
    elif len(tr.data) == 4501:
        #data.append(tr.data/max(tr.data))
        # data.append(tr.data[1000:-1]/max(tr.data[1000:-1]))
        #label.append(file_name)
        pass
    else:
        pass
print(count)
np.save('/home/sairaman/Desktop/stead-dataset/data/waveforms/trial_noise.npy', **dict(zip(label,data)))