import os
from modules import *

model_path = '../data/model/checkpoint_model_04'
base_dir = "../data/test_reshaped_images"
eval = Evaluation(model_path, base_dir, 300, 300, 32) #for ROC curve put batchsize = 1
eval.Confusion_matrix(fig_name='cm.png')


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

'''plt.figure(figsize=(10, 10))
class_names = test_ds.class_names
for images, labels in test_ds.take(1):
  for i in range(10,19):
    ax = plt.subplot(3, 3, i + 1-10)
    plt.imshow(images[i].numpy().astype("uint8"))
    print(labels[i])
    print(class_names,class_names[labels[i]])
    plt.title(class_names[labels[i]])
    plt.axis("off")
    plt.savefig("../figures/all_sample_figures.png")'''

'''#ROC curve
class_names = test_ds.class_names
print(test_ds.class_names)   
model = tf.keras.models.load_model("../data/model/trial")  

predictions = model.predict(test_ds).flatten()
test_labels = []
for x,y in test_ds:
    test_labels.append(y.numpy()[0])
print(test_labels)
print(predictions)
from sklearn.metrics import roc_curve
fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_labels, predictions)
from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras_model02 (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig("temp.png")
'''

'''# Confusion Matrix 
model = tf.keras.models.load_model("../data/model/trial_20per_data")
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
plt.savefig('../figures/trail_cm.png')'''

'''# load image
model = tf.keras.models.load_model("../data/model/trial")
base_dir = base_dir + 'EQ/'
file_list = os.listdir(base_dir)
class_names = test_ds.class_names
for file in file_list:
  print(file)
  image = tf.keras.utils.load_img(base_dir+file, target_size=(300, 300))
  input_arr = tf.keras.utils.img_to_array(image)
  input_arr = np.array([input_arr])  # Convert single image to a batch.
  predictions = model.predict(input_arr) #.argmax(axis = -1)
  print(predictions)'''