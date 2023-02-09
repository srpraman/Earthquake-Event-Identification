import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
base_dir = "../data/reshaped_images"
test_ds = image_dataset_from_directory(base_dir,
                                      validation_split=0.05,
                                      subset="validation",
                                      seed=123,
                                      image_size=(300, 300),
                                      batch_size=1)
print(test_ds.class_names)   
model = tf.keras.models.load_model("../data/model/model_all_param_20epochs")  
results = model.evaluate(test_ds)
print("TP,FP,TN,FN,ACC,PRECISION, RECALL, AUC")
TruePositives = results[1]
FalsePositives = results[2]
TrueNegatives= results[3]
FalseNegatives = results[4]
Recall = results[5]
print(results)
cm = np.array([[TruePositives, FalseNegatives],[FalsePositives, TrueNegatives]])
sns.heatmap(cm,annot=True)
plt.savefig('output.png')

'''#print(model.evaluate(test))                       
label = []
inp = []
for x,y in test_ds.as_numpy_iterator():
    label.append(y[0])
    inp.append(x)
    #print(y)
#print(np.array(inp)[:,0,...].shape)
label = np.array(label)
results = model.predict(np.array(inp)[:,0,...])
#results = results[:,0]
# print(results.shape,label.shape) 
print(results)
# cm  = confusion_matrix(label,results)
# print(cm)'''