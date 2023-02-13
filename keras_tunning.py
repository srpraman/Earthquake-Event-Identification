#########################################################
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
#############################################################
from tensorflow.data import AUTOTUNE
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Rescaling
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import image_dataset_from_directory, normalize
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import BinaryAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives, Precision, Recall, AUC
from tensorflow import keras
import sklearn
from sklearn.metrics import confusion_matrix
import seaborn as sns
####################################################
base_dir = "../data/reshaped_images"

train_ds = image_dataset_from_directory(base_dir,
                                        validation_split=0.2,
                                        subset="training",
                                        seed=123,
                                        image_size=(300, 300),
                                        batch_size=32)
val_ds = image_dataset_from_directory(base_dir,
                                      validation_split=0.2,
                                      subset="validation",
                                      seed=123,
                                      image_size=(300, 300),
                                      batch_size=32)
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

def build_model(hp):
    INPUT_SHAPE = (300, 300, 3)
    model = keras.Sequential([
    Rescaling(1./255, input_shape=INPUT_SHAPE),
    keras.layers.Conv2D(
        filters=hp.Int('conv_1_filter', min_value=16, max_value=128, step=16), 
        kernel_size=hp.Choice('conv_1_kernel', values = [3, 5]),
        activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(2, 2)),
  
    keras.layers.Conv2D( 
        filters=hp.Int('conv_2_filter', min_value=16, max_value=64, step=16),
        kernel_size=hp.Choice('conv_2_kernel', values = [3, 5]),
        activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(2, 2)),
    # adding flatten layer    
    keras.layers.Flatten(),
    keras.layers.BatchNormalization(),
    # adding dense layer    
    keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=8, max_value=64, step=16),
        activation='relu'
    ),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=8, max_value=64, step=16),
        activation='sigmoid'
    ),   
    keras.layers.Dense(1)
    ])
    #compilation of model
    metrics = [TruePositives(name='tp'), FalsePositives(name='fp'), TrueNegatives(name='tn'), FalseNegatives(name='fn'),
           BinaryAccuracy(name='accuracy'), Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model
"""model = Sequential([
    Rescaling(1./255, input_shape=INPUT_SHAPE),
    Conv2D(16, 5, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(5, 5), strides=(2, 2)),
    Conv2D(32, 5, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(5, 5), strides=(2, 2)),
    Flatten(),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='sigmoid'),
    Dense(1)])

metrics = [TruePositives(name='tp'), FalsePositives(name='fp'), TrueNegatives(name='tn'), FalseNegatives(name='fn'),
           BinaryAccuracy(name='accuracy'), Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]
model.compile(loss='binary_crossentropy',
              optimizer='adam',            
              metrics=metrics)"""
from keras_tuner import RandomSearch
#creating randomsearch object
tuner = RandomSearch(build_model,
                    objective='val_accuracy',
                    max_trials = 30)
# search best parameter
tuner.search(train_ds,epochs=10,validation_data=val_ds)
print(tuner.results_summary())

model=tuner.get_best_models(num_models=1)[0]

model.save('../data/model/tuning_10_epochs.model')
model = keras.models.load_model("../data/model/tuning_10_epochs.model")
#print(model.evaluate(test_ds))
print(model.summary())