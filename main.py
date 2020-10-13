from keras.datasets import cifar10

from keras.utils import np_utils #to one-hot
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from PIL import Image
from keras.preprocessing.image import img_to_array
np.random.seed(10)


# Modify the following function to show more images in a window
categorical={
0: 'airplane',
1: 'automobile',
2: 'bird',
3: 'cat',
4: 'deer',
5: 'dog',
6: 'frog',
7: 'horse',
8: 'ship',
9: 'truck'}

def plot_image(images, labels,predict):
	
	fig = plt.gcf()
	plt.imshow(np.reshape(images, (32, 32,3)), cmap='binary')
	plt.title("Label= "+categorical[int(labels)]+',\n'+"Predict= "+categorical[int(predict)])

x_number=random.sample(range(10000),15)
  
batch_size 	=128
num_classes = 10
epochs 		= 30 


(x_train_image,y_train_label),(x_test_image, y_test_label)=cifar10.load_data()
x_train= x_train_image.reshape(x_train_image.shape[0],32,32,3).astype('float32')
x_test = x_test_image.reshape(x_test_image.shape[0],32,32,3).astype('float32')

#normalization
x_train_norm=x_train/255
x_test_norm=x_test/255

#ans one hot
y_train_onehot=np_utils.to_categorical(y_train_label)
y_test_onehot=np_utils.to_categorical(y_test_label)

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),padding='same', 
                 input_shape=(32,32,3),
                 strides=(1,1),activation='relu'))
model.add(Conv2D(32,kernel_size=(3,3),padding='same', strides=(1,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu',padding='same'))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())

model.add(Dense(400, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(400, input_dim=784))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
#from keras.optimizers import rmsprop
#opt = rmsprop(lr=0.0001, decay=1e-6)


model.summary()
# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])



history=model.fit(x=x_train_norm,
                  y=y_train_onehot,validation_split=0, 
                  epochs=epochs, batch_size=batch_size,verbose=1,
                  validation_data=(x_test_norm,y_test_onehot))


model.save('cifar-10.h5')
score = model.evaluate(x_test_norm, y_test_onehot, verbose=0)
results = model.predict_classes(x_test_norm)

'''cifar-test'''
plt.figure(figsize=(20,18))

for i,j in zip(x_number,range(15)):
    plt.subplot(3,5,j+1)
    plot_image(x_test_norm[i],y_test_label[i],results[i])
    
plt.suptitle("Test loss: %f, Test accuracy= %f" %(score[0],score[1]),fontsize=18)
plt.tight_layout()
plt.savefig('Keras Cifar-10 model.png',dpi=300,format='png')
print("Keras Cifar-10 model.png is saved")
plt.show()

'''self-cifar-test'''
x_self=np.array(range(10))
np.random.shuffle(x_self)
self_pred=[]
plt.figure(figsize=(20,10))
for i in range(10):
  img = Image.open('picture/'+categorical[int(x_self[i])]+'.png')
  img_convert_ndarray = np.array(img)
  ndarray_convert_img = np.array(img)
  x_selftest=ndarray_convert_img[0:32,0:32,0:3]
  x_selftest = x_selftest.reshape(1, 32, 32, 3).astype('float32')
  x_selftest /= 255
  self_pred.append(model.predict_classes(x_selftest,verbose=0))
  plt.subplot(2,5,i+1)
  plot_image(x_selftest,x_self[i],self_pred[i])

plt.tight_layout()
plt.savefig('Keras Self Cifar-10 model.png',dpi=300,format='png')
print("Keras Self Cifar-10 model.png is saved")
plt.show()
