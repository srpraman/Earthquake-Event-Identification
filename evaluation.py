import pandas as pd
import os
from modules import *

# model_path = '../data/model/checkpoint_model_05'
# base_dir = "../data/test_reshaped_images"
# eval = Evaluation(model_path, base_dir, 300, 300, 1) #for ROC curve put batchsize = 1
# eval.Confusion_matrix(fig_name='checkpoint_model_05/checkpoint_model_05.png')


'''base_dir = "../data/aug_test_reshaped_images"
# base_dir = "../../trial_folder/reshaped_seismograms/"
#base_dir = "../../filter/NOISE_spec_reshape/"
test_ds = tf.keras.utils.image_dataset_from_directory(base_dir, 
                                      seed = 123,
                                      image_size = (128, 128),
                                      batch_size = 64)'''

'''test_labels = []
for x,y in test_ds:
    test_labels.append(y.numpy()[0])
np.savez_compressed('../data/evaluation/checkpoint_model_04/test_labels.npz', np.array(test_labels))
model = tf.keras.models.load_model("../data/model/trial")  
predictions = model.predict(test_ds).flatten()
np.savez_compressed('../data/evaluation/checkpoint_model_04/predictions.npz', np.array(predictions))
print(np.load("../data/evaluation/checkpoint_model_04/test_labels.npz")['arr_0'].shape)
print(np.load("../data/evaluation/checkpoint_model_04/predictions.npz")['arr_0'].shape)'''


# load image
model = tf.keras.models.load_model("../data/model/checkpoint_model_04")
base_dir = '../data/evaluation/reshape_spec_files/'
#base_dir = '../data/test_reshaped_images/EQ/'

file_list = os.listdir(base_dir)
# class_names = test_ds.class_names
pred = []
file_name = []
for file in file_list:
  image = tf.keras.utils.load_img(base_dir + file, target_size=(300, 300))
  input_arr = tf.keras.utils.img_to_array(image)
  input_arr = np.array([input_arr])  # Convert single image to a batch.
  predictions = model.predict(input_arr, verbose=False) #.argmax(axis = -1)
  # print(predictions)
  if predictions[0][0] < 0.5:
    # print(file)
    # pred.append(predictions[0][0])
    # file_name.append(file)
    # print(predictions[0,0])
    print(file, "---->", predictions[0, 0])
# print(file, pred)   