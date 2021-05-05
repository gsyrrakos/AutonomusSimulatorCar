# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from kerastuner.engine.hyperparameters import Choice, HyperParameters
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from Preprocces import *

path = 'MyData'
data = ImportDataInfo(path)

balanceData(data)

imagesPath, steerings = loadData(path, data)

xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2, random_state=10)
print('Total Training Images: ', len(xTrain))
print('Total Validation Images: ', len(xVal))

print('Setting UP')
import os
import kerastuner as kt



hp = HyperParameters()
# This will override the `learning_rate` parameter with your
# own selection of choices
hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])


def createModel(hp):
    model = Sequential()

    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), loss='mse')
    return model


def build_model(hp):
    inputs = tf.keras.Input(shape=(66, 200, 3))
    x = inputs
    for i in range(hp.Int('conv_blocks', 3, 5, default=3)):
        filters = hp.Int('filters_' + str(i), 24, 128, step=32)
        for _ in range(2):
            x = tf.keras.layers.Convolution2D(
                filters, kernel_size=(3, 3), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ELU()(x)
        if hp.Choice('pooling_' + str(i), ['avg', 'max']) == 'max':
            x = tf.keras.layers.MaxPool2D()(x)
        else:
            x = tf.keras.layers.AvgPool2D()(x)
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dense(
        hp.Int('hidden_size', 30, 100, step=10, default=50),
        activation='elu')(x)
    x = tf.keras.layers.Dropout(
        hp.Float('dropout', 0, 0.5, step=0.1, default=0.5))(x)
    outputs = tf.keras.layers.Dense(10, activation='elu')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
        loss='mse')
    return model


tuner = kt.Hyperband(createModel,
                     hyperparameters=hp,
                     objective='val_loss',
                     max_epochs=10,
                     tune_new_entries=False,
                     directory='my_dir',
                     project_name='intro_to_kt')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(batchGen(xTrain, yTrain, 64, 1), epochs=20, steps_per_epoch=300,
             validation_data=batchGen(xVal, yVal, 64, 0), validation_steps=300, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(batchGen(xTrain, yTrain, 64, 1), epochs=20, steps_per_epoch=300,
                    validation_data=batchGen(xVal, yVal, 64, 0), validation_steps=300, callbacks=[stop_early])

val_acc_per_epoch = history.history['val_loss']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

# train_data = create_data_batches(xTrain,yTrain)
# val_data = create_data_batches(xVal,yVal, valid_data=True)

'''
model = createModel()
model.summary()
history = model.fit(batchGen(xTrain, yTrain, 256, 1), steps_per_epoch=300, epochs=20,
                    validation_data=batchGen(xVal, yVal, 256, 0), validation_steps=300)
model.save('model.h5')
print('model saved')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0, 1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
'''