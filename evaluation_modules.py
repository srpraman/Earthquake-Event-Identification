import tensorflow as tf
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

class Evaluation:
    def __init__(self, model_path, data_path, height, length, batch_size):
        self.model = tf.keras.models.load_model(model_path)
        self.data = tf.keras.utils.image_dataset_from_directory(data_path, 
                                      seed = 123,
                                      image_size = (height, length),
                                      batch_size = batch_size)
    
    def Confusion_matrix(self, fig_name='cm.png'):
        self.results = self.model.evaluate(self.data)
        print("TP,FP,TN,FN,ACC,PRECISION, RECALL, AUC")
        self.TruePositives = self.results[1]
        self.FalsePositives = self.results[2]
        self.TrueNegatives= self.results[3]
        self.FalseNegatives = self.results[4]
        self.Recall = self.results[7]
        self.cm = np.array([[self.TruePositives, self.FalseNegatives],[self.FalsePositives, self.TrueNegatives]])
        categories = ["Noise","Earthquake"]
        sns.heatmap(self.cm,annot=True,xticklabels=categories,yticklabels=categories)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        name = "../figures/" + fig_name
        plt.savefig(name)
    
    def Sample_figures(self, fig_name='all_sample_figures.png'):
        plt.figure(figsize=(10, 10))
        self.class_names = self.data.class_names
        for images, labels in self.data.take(1):
            for i in range(10,19):
                ax = plt.subplot(3, 3, i + 1-10)
                plt.imshow(images[i].numpy().astype("uint8"))
                print(labels[i])
                print(self.class_names, self.class_names[labels[i]])
                plt.title(self.class_names[labels[i]])
                plt.axis("off")
                name = "../figures/" + fig_name
                plt.savefig(name)
                
    def ROC_curve(self, fig_name="roc.png"):
        self.class_names = self.data.class_names
        predictions = self.model.predict(self.data).flatten()
        test_labels = []
        for x,y in self.data:
           test_labels.append(y.numpy()[0])
        # print(test_labels)
        # print(predictions)
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_labels, predictions)
        auc_keras = auc(fpr_keras, tpr_keras)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_keras, tpr_keras, label='Keras_model02 (area = {:.3f})'.format(auc_keras))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        name = "../figures/" + fig_name
        plt.savefig(name)
        
    