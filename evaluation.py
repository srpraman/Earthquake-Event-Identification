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


#ROC curve
class_names = test_ds.class_names
print(test_ds.class_names)   
model = tf.keras.models.load_model("../data/model/checkpoint_model_01")  
#probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = model.predict(test_ds).flatten()
test_labels = []
for x,y in test_ds:
    test_labels.append(y.numpy()[0])

from sklearn.metrics import roc_curve
fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_labels, predictions)
from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)
###
model = tf.keras.models.load_model("../data/model/checkpoint_model_02")  
predictions = model.predict(test_ds).flatten()
test_labels = []
for x,y in test_ds:
    test_labels.append(y.numpy()[0])

from sklearn.metrics import roc_curve
fpr_keras2, tpr_keras2, thresholds_keras2 = roc_curve(test_labels, predictions)
auc_keras2 = auc(fpr_keras, tpr_keras)
##
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras_model01 (area = {:.3f})'.format(auc_keras))
plt.plot(fpr_keras2, tpr_keras2, label='Keras_model02 (area = {:.3f})'.format(auc_keras2))

#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig("temp.png")


#Confusion Matrix 
'''model = tf.keras.models.load_model("../data/model/checkpoint_model_01")
results = model.evaluate(test_ds)
print("TP,FP,TN,FN,ACC,PRECISION, RECALL, AUC")
TruePositives = results[1]
FalsePositives = results[2]
TrueNegatives= results[3]
FalseNegatives = results[4]
Recall = results[7]
cm = np.array([[TruePositives, FalseNegatives],[FalsePositives, TrueNegatives]])
categories = ["Noise","Earthquake"]
sns.heatmap(cm,annot=True,xticklabels=categories,yticklabels=categories)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('../figures/checkpoint_model_02/checkpoint_model_01_checker_board.png')'''
