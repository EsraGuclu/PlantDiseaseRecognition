from google.colab import drive
drive.mount('/content/drive')

import numpy as np 
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras import applications  
from tensorflow.keras.utils import to_categorical  
import matplotlib.pyplot as plt  
import math

img_width, img_height = 224, 224  

model_weights = '/content/drive/MyDrive/Colab Notebooks/tf_bean_dataset_final/vgg_model.h5' 
#model_weights = '/content/drive/MyDrive/Colab Notebooks/tf_bean_dataset_final/mobilenet_model.h5' 
#model_weights = '/content/drive/MyDrive/Colab Notebooks/tf_bean_dataset_final/resnet_model.h5' 

#dataset directories 
train_data_dir = '/content/drive/MyDrive/Colab Notebooks/tf_bean_dataset_final/train'  
validation_data_dir = '/content/drive/MyDrive/Colab Notebooks/tf_bean_dataset_final/validation'  

epochs = 50  
batch_size = 32

#build network without last layer/top layer
model = applications.VGG16(include_top=False, weights='imagenet') 
#model = applications.MobileNetV2(include_top=False, weights='imagenet') 
#model = applications.ResNet50V2(include_top=False, weights='imagenet')

#Add data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   fill_mode='nearest')

#Flow training images using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode=None,
                                                    shuffle=False )

#get size of the training set
train_samples = len(train_generator.filenames)
#get class number
num_classes = len(train_generator.class_indices)

predict_size_train = int(math.ceil(train_samples / batch_size))
bean_features_train = model.predict_generator(train_generator, predict_size_train)

#save trainig set features
np.save('/content/drive/MyDrive/Colab Notebooks/tf_bean_dataset_final/vgg_features_train.npy', bean_features_train)
#np.save('/content/drive/MyDrive/Colab Notebooks/tf_bean_dataset_final/mobilenet_features_train.npy', bean_features_train)
#np.save('/content/drive/MyDrive/Colab Notebooks/tf_bean_dataset_final/resnet_features_train.npy', bean_features_train)

val_datagen = ImageDataGenerator(rescale=1. / 255)

val_generator = val_datagen.flow_from_directory(validation_data_dir,
                                                target_size=(img_width, img_height),
                                                batch_size=batch_size,
                                                class_mode=None,
                                                shuffle=False )

#get size of the validation set
validation_samples = len(val_generator.filenames)
#get class number
predict_size_validation = int(math.ceil(validation_samples / batch_size))

bean_features_validation = model.predict_generator(val_generator, predict_size_validation)

#save validation set features
np.save('/content/drive/MyDrive/Colab Notebooks/tf_bean_dataset_final/vgg_features_val.npy', bean_features_validation)
#np.save('/content/drive/MyDrive/Colab Notebooks/tf_bean_dataset_final/mobilenet_features_val.npy', bean_features_validation)
#np.save('/content/drive/MyDrive/Colab Notebooks/tf_bean_dataset_final/resnet_features_val.npy', bean_features_validation)'''

# load features saved earlier  
train_data = np.load('/content/drive/MyDrive/Colab Notebooks/tf_bean_dataset_final/vgg_features_train.npy')  
#train_data = np.load('/content/drive/MyDrive/Colab Notebooks/tf_bean_dataset_final/mobilenet_features_train.npy')  
#train_data = np.load('/content/drive/MyDrive/Colab Notebooks/tf_bean_dataset_final/resnet_features_train.npy')  

# get the class labels for the training data, in the original order  
train_labels = train_generator.classes  
# convert the training labels to categorical vectors  
train_labels = to_categorical(train_labels, num_classes=num_classes)  

val_data = np.load('/content/drive/MyDrive/Colab Notebooks/tf_bean_dataset_final/vgg_features_val.npy')  
#val_data = np.load('/content/drive/MyDrive/Colab Notebooks/tf_bean_dataset_final/mobilenet_features_val.npy')
#val_data = np.load('/content/drive/MyDrive/Colab Notebooks/tf_bean_dataset_final/resnet_features_val.npy')

val_labels = val_generator.classes  
val_labels = to_categorical(val_labels, num_classes=num_classes)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.999):
      print("\nReached 99.9% accuracy so cancelling training!")
      self.model.stop_training = True

#create and train network
final_model = Sequential()  
final_model.add(Flatten(input_shape=train_data.shape[1:])) 
final_model.add(Dense(256, activation='relu'))
final_model.add(Dropout(0.5))
final_model.add(Dense(128, activation='relu'))
final_model.add(Dropout(0.2))
final_model.add(Dense(64, activation='relu'))
final_model.add(Dropout(0.2))
final_model.add(Dense(num_classes, activation='softmax'))  



final_model.compile(optimizer='adam',  
                    loss='binary_crossentropy', 
                    metrics=['accuracy'] )  

callbacks = myCallback()
history = final_model.fit(train_data, 
                          train_labels,  
                          epochs=epochs,  
                          batch_size=batch_size,  
                          validation_data = (val_data, val_labels),
                          verbose = 1,
                          callbacks=[callbacks])  

#final_model.summary()

final_model.save_weights(model_weights)  

(loss, acc) = final_model.evaluate(val_data, 
                                  val_labels, 
                                  batch_size=batch_size, 
                                  verbose=1)

print("[INFO] accuracy: {:.2f}%".format(acc * 100))  
print("[INFO] Loss: {}".format(loss))

plt.figure(1)  

# summarize for accuracy  
plt.subplot(211)  
plt.plot(history.history['accuracy'])  
plt.plot(history.history['val_accuracy'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'val'], loc='upper left')  
plt.show() 

# summarize history for loss  
plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'val'], loc='upper left')  
plt.show()

