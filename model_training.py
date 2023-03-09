#########################################################
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
####################################################

def data_creation(base_dir, length, height):
    train_ds = image_dataset_from_directory(base_dir,
                                                validation_split=0.8,
                                                subset="training",
                                                seed=123,
                                                image_size=(length, height),
                                                batch_size=32)
    val_ds = image_dataset_from_directory(base_dir,
                                            validation_split=0.8,
                                            subset="validation",
                                            seed=123,
                                            image_size=(length, height),
                                            batch_size=32)
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

def create_model(INPUT_SHAPE):
    model = Sequential([
        Rescaling(1./255, input_shape=INPUT_SHAPE),
        Conv2D(8, 5, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(5, 5), strides=(2, 2)),
        Conv2D(16, 5, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(5, 5), strides=(2, 2)),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='sigmoid'),
        Dense(1, activation='sigmoid')])

    metrics = [TruePositives(name='tp'), FalsePositives(name='fp'), TrueNegatives(name='tn'), FalseNegatives(name='fn'),
            BinaryAccuracy(name='accuracy'), Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]
    model.compile(loss='binary_crossentropy',
                optimizer='adam',            
                metrics=metrics)
    return model

base_dir = "../data/aug_reshaped_images"
length = 128
height = 128
layers = 3
train_ds, val_ds = data_creation(base_dir, length, height)
model = create_model((length, height, layers))
print(model.summary())    
###############################################################  
epochs=20
checkpoint_filepath = "../data/model/trial_20per_data"
model_checkpoint_callback = ModelCheckpoint(
                            filepath=checkpoint_filepath,
                            save_freq='epoch',
                            monitor='val_accuracy',
                            mode='max',
                            save_best_only=True)

history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[model_checkpoint_callback])
##############################################################
# # Performance Graph
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
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
plt.show()
plt.savefig("../figures/trial_20per_data.png") # checkpoint_model_03/checkpoint_model_03.png")