import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# base_dir = "../data/test_reshaped_images"
base_dir = "../../trial_folder/reshaped_seismograms"
test_ds = image_dataset_from_directory(base_dir, 
                                    seed = 123,
                                    image_size = (300, 300),
                                    batch_size = 1)

class_names=test_ds.class_names
print("class_names", class_names)
"""plt.figure(figsize=(10, 10))
for images, labels in test_ds.take(1):
  for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    print(labels[i])
    print(class_names,class_names[labels[i]])
    plt.title(class_names[labels[i]])
    plt.axis("off")"""
    #plt.savefig("trial.png")
'''class_names = test_ds.class_names
print(test_ds.class_names)   
model = tf.keras.models.load_model("../data/model/model_all_param_20epochs_separate_testdata")  
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_ds)
print("prediction: ", predictions)
test_labels = []
for x,y in test_ds:
    test_labels.append(y.numpy()[0])
print("label: ", test_labels[0])'''

#Confusion Matrix and Classification Report
model = tf.keras.models.load_model("../data/model/checkpoint_model_02.hdf5")
results = model.evaluate(test_ds)
print(results)
print(f"TP: {results[1]},FP: {results[2]},TN: {results[3]},FN: {results[4]},ACC: {results[5]},PRECISION: {results[6]}, RECALL: {results[7]}, AUC: {results[8]}")
'''results = model.evaluate(test_ds)
print("TP,FP,TN,FN,ACC,PRECISION, RECALL, AUC")
TruePositives = results[1]
FalsePositives = results[2]
TrueNegatives= results[3]
FalseNegatives = results[4]
Recall = results[7]
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