import matplotlib.pyplot as plt
from tensorflow.data import AUTOTUNE
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Rescaling
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import image_dataset_from_directory, normalize
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import BinaryAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives, Precision, Recall, AUC
from tensorflow import keras
import h5py

class CNN:
    def __init__(self, data_path, height, length, batch_size):
        self.length = length
        self.height = height
        self.train_ds = image_dataset_from_directory(data_path,
                                                validation_split=0.2,
                                                subset="training",
                                                seed=123,
                                                image_size=(self.length, self.height),
                                                batch_size=batch_size)
        self.val_ds = image_dataset_from_directory(data_path,
                                            validation_split=0.2,
                                            subset="validation",
                                            seed=123,
                                            image_size=(self.length, self.height),
                                            batch_size=batch_size)
        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    def Create_model(self, filter_size=5, pool=5, stride=2, dropout=0.2):
        self.input_shape = (self.length , self.height, 3)
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