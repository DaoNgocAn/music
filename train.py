from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, Callback
from keras.models import load_model
from create_model import create_model

import time
import numpy as np
import os
import argparse
from config import Config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


parser = argparse.ArgumentParser()
parser.add_argument(
        '-b', '--batch_size', dest = 'batch_size',
        type = int, default=16,
        help = 'Batch size')
parser.add_argument(
    '--use_pretrain_model', dest='use_pretrain_model',
    type=bool, default=False,
    help='load pretrain model'
)

args = parser.parse_args()
def get_time(localtime):
    return '{}_{}'.format(localtime.tm_mday, localtime.tm_mon)
model_file = 'model_{}.h5'.format(get_time(time.localtime(time.time())))
log_file = './log_' + time.asctime(time.localtime(time.time()))[8:]

model_name = 'DenseNet'
if args.use_pretrain_model:
    model = load_model('models/model_DensenNet.h5')
else:
    model = create_model(model_name, (Config.IMG_W, Config.IMG_H, 1))
model.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(lr=1e-4), metrics=["accuracy"])
batch_size = args.batch_size

class DataGenerator(object):
    def __init__(self, dim_x=129, dim_y=129, batch_size=32, shuffle=True):
        'Initialization'
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def generate(self, X, y):
        def __data_generation(list_IDs):
            # Initialization
            _X = np.empty((self.batch_size, self.dim_x, self.dim_y, 1))
            _y = np.empty((self.batch_size, 10), dtype=int)

            # Generate data
            for i, ID in enumerate(list_IDs):
                # Store volume
                _X[i, :, :, 0] = np.load(X[ID])
                # Store class
                _y[i] = y[ID]
            return _X, _y

        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(len(X))

            # Generate batches
            imax = int(len(indexes) / self.batch_size)
            for i in range(imax):
                # Generate data
                _X, _y = __data_generation(indexes[i * self.batch_size:(i + 1) * self.batch_size])

                yield _X, _y

    def __get_exploration_order(self, number_samples):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(number_samples)
        if self.shuffle:
            np.random.shuffle(indexes)
        return indexes



X_train = []
y_train = []
X_val = []
y_val = []

slices_path = Config.NPY_SLICES_PATH

for folder in os.listdir(slices_path):
    for label in os.listdir(os.path.join(slices_path, folder)):
        for filename in os.listdir(os.path.join(slices_path, folder, label)):
            _y = np.zeros(Config.NB_CLASSES)
            _y[int(label) - 1] = 1
            if folder == 'train':
                X_train.append(os.path.join(slices_path, folder, label, filename))
                y_train.append(_y)
            else:
                X_val.append(os.path.join(slices_path, folder, label, filename))
                y_val.append(_y)

# np.random.seed(20140002)
# print len(X_train)
# data_train = [i for i in zip(X_train, y_train) if np.random.random() > 0.999]
# for d in data_train:
#     print d
# X_train, y_train = zip(*data_train)
# X_train = X_train[:2] + X_train[-2:]
# y_train = y_train[:2] + y_train[-2:]
# print y_train
# X_train = np.array([np.load(i).reshape(img_w, img_h, 1) for i in X_train])
# print X_train.shape
# model.fit(X_train, np.array(y_train), epochs=500000, callbacks=[tensorboard])

train_generator = DataGenerator(Config.IMG_W, Config.IMG_H, batch_size, shuffle=True).generate(X_train, y_train)
val_generator = DataGenerator(Config.IMG_W, Config.IMG_H, batch_size, shuffle=True).generate(X_val, y_val)

checkpoint = ModelCheckpoint(model_file, monitor='val_acc', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)

early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
tensorboard = TensorBoard(log_dir=log_file, histogram_freq=0, batch_size=32, write_graph=True,
                          write_grads=False, write_images=False, embeddings_freq=0,
                          embeddings_layer_names=None, embeddings_metadata=None)


class Evaluate(Callback):
    def __init__(self, generator, steps):
        super(Evaluate, self).__init__()
        self.generator = generator
        self.steps = steps

    def on_epoch_end(self, batch, logs={}):
        print('test' + str(self.model.evaluate_generator(self.generator, steps=self.steps)[1]))


print(len(X_val), len(X_train))
print("training ...")
model.fit_generator(generator=train_generator,
                    steps_per_epoch=len(X_train) // batch_size,
                    validation_data=val_generator,
                    epochs=100,
                    validation_steps=len(X_val) // batch_size,
                    callbacks=[tensorboard, checkpoint, early],
                    verbose=2
                    )
# --------------------------------------------------------------------------------
print("*" * 79 + "\n")

del model
model_file = 'model_{}.h5'.format(get_time(time.localtime(time.time())))
log_file = './log_' + time.asctime(time.localtime(time.time()))[8:]

model_name = 'ResNet50'
if args.use_pretrain_model:
    model = load_model('models/model_ResNet50.h5')
else:
    model = create_model(model_name, (Config.IMG_W, Config.IMG_H, 1))
model.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(lr=1e-4), metrics=["accuracy"])
checkpoint = ModelCheckpoint(model_file, monitor='val_acc', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)

early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
tensorboard = TensorBoard(log_dir=log_file, histogram_freq=0, batch_size=32, write_graph=True,
                          write_grads=False, write_images=False, embeddings_freq=0,
                          embeddings_layer_names=None, embeddings_metadata=None)
model.fit_generator(generator=train_generator,
                    steps_per_epoch=len(X_train) // batch_size,
                    validation_data=val_generator,
                    epochs=100,
                    validation_steps=len(X_val) // batch_size,
                    callbacks=[tensorboard, checkpoint, early],
                    verbose=2
                    )

