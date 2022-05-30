import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold

# Load FashionMNIST data set
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Using only a part of the test set (first of 10 folds)
# dividing into 6 sets randomly To get generalised data
skf = StratifiedKFold(n_splits=6, random_state=0, shuffle=False)  
for train_index, test_index in skf.split(x_train, y_train):
    x_train, y_train = x_train[test_index], y_train[test_index]
    break
# labeling categorical values
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#################################################################################
# Training the model

x_train = x_train.reshape(10000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Randomly chosen
batch_size = 128
num_classes = 10
# no of times step function (Here relu)
epochs = 40

#Regularization - used for inc accuracy
#Seq model (used for image classification)
model = Sequential()

#Neural Network
#layers
#transforming function - relu
model.add(Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=x_train.shape[1:]))
model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.5))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(rate=0.5))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
#batch normaliztion helps in restricting the mean in the range of 0 to 1 
#Dropout func helps us to recognize whether the img is more classified (Deciding by comapring to rate = 0.75)
model.add(BatchNormalization())  
model.add(Dropout(rate=0.75))  
model.add(Dense(num_classes, activation='softmax'))

# RMS propagation - Algo
opt = keras.optimizers.rmsprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Normalization
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Learning rate reduction - Algo
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          verbose=1,
          callbacks=[learning_rate_reduction])

# Results on test set
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

#################################################################################

# Save the model to a file
model.save('model.h5')