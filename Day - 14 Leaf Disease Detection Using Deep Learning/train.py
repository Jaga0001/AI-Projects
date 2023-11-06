from keras.models import Sequential

from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

#Basic CNN

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)))
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
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   rotation_range=20,  # Example: Adding rotation augmentation
                                   width_shift_range=0.2,  # Example: Shifting width of images
                                   height_shift_range=0.2  # Example: Shifting height of images
                                   )

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('train',
                                              target_size=(128, 128),
                                              batch_size=32,
                                              class_mode='categorical')


labels = (train_set.class_indices)
print(labels)

test_set = test_datagen.flow_from_directory('val',
                                            target_size=(128, 128),
                                            batch_size=32,
                                            class_mode='categorical')
labels2 = (test_set.class_indices)
print(labels2)


model.fit(train_set,
          steps_per_epoch=375,
          epochs=10,
          validation_data=test_set,
          validation_steps=125)

model_json = model.to_json()
with open('model.json','w') as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved Model to Disk")
