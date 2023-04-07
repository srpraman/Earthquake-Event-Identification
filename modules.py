import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.data import AUTOTUNE
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Rescaling
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import image_dataset_from_directory, normalize
from keras.callbacks import ModelCheckpoint
from keras.metrics import BinaryAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives, Precision, Recall, AUC
import h5py
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os
import seaborn as sns

class CNN:
    def __init__(self, data_path, height, length, batch_size):
        self.length = length
        self.height = height
        self.train_ds = image_dataset_from_directory(data_path,
                                                validation_split=0.2,
                                                subset="training",
                                                seed=123,
                                                image_size=(self.length, self.height),
                                                batch_size=batch_size,
                                                color_mode='grayscale')
        self.val_ds = image_dataset_from_directory(data_path,
                                            validation_split=0.2,
                                            subset="validation",
                                            seed=123,
                                            image_size=(self.length, self.height),
                                            batch_size=batch_size,
                                            color_mode='grayscale')
        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
    def Create_model(self, filter_size=5, pool=5, stride=2, dropout=0.2):
        self.input_shape = (self.length, self.height, 1)
        self.model = Sequential([
            Rescaling(1./255, input_shape=self.input_shape),
            Conv2D(8, filter_size, padding='same', activation='relu', name='layer1'),
            MaxPooling2D(pool_size=(pool, pool), strides=(stride, stride), name='layer2'),
            Conv2D(16, filter_size, padding='same', activation='relu', name='layer3'),
            MaxPooling2D(pool_size=(pool, pool), strides=(stride, stride), name='layer4'),
            Flatten(),
            Dense(32, activation='relu', name='layer5'),
            Dropout(dropout),
            Dense(16, activation='sigmoid', name='layer6'),
            Dense(1, activation='sigmoid', name='layer7')])      
        metrics = [TruePositives(name='tp'), FalsePositives(name='fp'), TrueNegatives(name='tn'), FalseNegatives(name='fn'),
                BinaryAccuracy(name='accuracy'), Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]
        self.model.compile(loss='binary_crossentropy',
                    optimizer='adam',            
                    metrics=metrics)
        print(self.model.summary())
        return self.model 
   
    def Train_model(self, epochs, checkpoint_filepath):
        self.epochs = epochs
        model_checkpoint_callback = ModelCheckpoint(
                            filepath=checkpoint_filepath,
                            save_freq='epoch',
                            monitor='val_accuracy',
                            mode='max',
                            save_best_only=True)
        self.history = self.model.fit(self.train_ds, validation_data=self.val_ds, epochs=self.epochs, callbacks=[model_checkpoint_callback])          
   
    def Training_graph(self, fig_name='training_plot.png'):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(self.epochs)
        print(epochs_range)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        name = "../figures/" + fig_name
        plt.savefig(name)
        # plt.savefig("../figures/cp2_with_layer_names.png") # checkpoint_model_03/checkpoint_model_03.png")
        

class Evaluation:
    def __init__(self, model_path, data_path, height, length, batch_size):
        self.model = tf.keras.models.load_model(model_path)
        self.data = tf.keras.utils.image_dataset_from_directory(data_path, 
                                                                seed = 123,
                                                                image_size = (height, length),
                                                                batch_size = batch_size,
                                                                color_mode='grayscale')
        self.class_names = self.data.class_names
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
    
    