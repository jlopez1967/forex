# importing libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K

###https://www.geeksforgeeks.org/python-image-classification-using-keras/
###https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
img_width, img_height = 224, 224
img_width, img_height = 480, 349

train_data_dir = 'v_data_a2/train'
validation_data_dir = 'v_data_a2/valid'
nb_train_samples = 3500
nb_validation_samples = 1500
epochs = 80
batch_size = 25

if K.image_data_format() == 'channels_first':
	input_shape = (3, img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)

    
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.3))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
  
    

model.compile(loss ='binary_crossentropy',
					optimizer ='rmsprop',
				metrics =['accuracy'])

train_datagen = ImageDataGenerator(
				rescale = 1. / 255,
				shear_range = 0.2,
				zoom_range = 0.2,
			horizontal_flip = True)


test_datagen = ImageDataGenerator(rescale = 1. / 255)



train_generator = train_datagen.flow_from_directory(train_data_dir,
							target_size =(img_width, img_height),
					batch_size = batch_size, class_mode ='binary',shuffle=False)

validation_generator = test_datagen.flow_from_directory(
									validation_data_dir,
				target_size =(img_width, img_height),
		batch_size = batch_size, class_mode ='binary',shuffle=False)


model.fit_generator(train_generator,
	steps_per_epoch = nb_train_samples // batch_size,
	epochs = epochs, validation_data = validation_generator,
	validation_steps = nb_validation_samples // batch_size)

model.save_weights('model_saved.h5')


