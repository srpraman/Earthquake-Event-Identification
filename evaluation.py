import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
base_dir = "../data/test_reshaped_images"
test_ds = image_dataset_from_directory(base_dir, 
                                    seed = 123,
                                    image_size = (300, 300),
                                    batch_size = 1)
class_names = test_ds.class_names
print(test_ds.class_names)   
model = tf.keras.models.load_model("../data/model/model_all_param_20epochs_separate_testdata")  
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_ds)
print("prediction: ", predictions)
test_labels = []
for x,y in test_ds:
    test_labels.append(y.numpy()[0])
print("label: ", test_labels[0])

#Confusion Matrix and Classification Report
'''results = model.evaluate(test_ds)
print("TP,FP,TN,FN,ACC,PRECISION, RECALL, AUC")
TruePositives = results[1]
FalsePositives = results[2]
TrueNegatives= results[3]
FalseNegatives = results[4]
Recall = results[5]
print(results)
cm = np.array([[TruePositives, FalseNegatives],[FalsePositives, TrueNegatives]])
sns.heatmap(cm,annot=True)
plt.savefig('output_with_separate_testdata.png')'''

'''from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
Y_pred = model.predict(val_ds)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(y, y_pred))
print('Classification Report')
target_names = ['EARTHQ','NOISE']
print(classification_report(y, y_pred, target_names=target_names))
cm = confusion_matrix(y,y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()'''