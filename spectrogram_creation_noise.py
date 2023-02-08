
import os
import obspy
import matplotlib.pyplot as plt

file_list = os.listdir("/home/sairaman/Desktop/stead-dataset/data/waveforms/chunk1_ir_removed")
file_list = file_list[10000:]

#os.chdir("/mnt/home/chunk2_images")
def magnitude(file_name,df):
    sl_no = int(file_name.split("_")[-1].split(".")[0])

def spec_plot(file_list):
  
    for file_name in file_list:
        try:   
            st = obspy.read("/home/sairaman/Desktop/stead-dataset/data/waveforms/chunk1_ir_removed/" + file_name)
            st.detrend("linear")
            st.filter('bandpass', freqmin=2, freqmax=10, corners=4, zerophase=True)
            st.spectrogram(log=True,outfile="/home/sairaman/Desktop/stead-dataset/data/images/NOISE/" + file_name.split(".")[0] +'.jpeg',fmt='jpeg',show=False)
            plt.plot()
            plt.close() 
            #obspy.spectrogram(st,log=True,outfile="/home/sairaman/Desktop/stead-dataset/data/images/EQ/" + file_name.split(".")[0] +'.jpeg',fmt='jpeg')
            #st.plot()
        except:
            print(f"error in file {file_name}")
            pass
spec_plot(file_list) 