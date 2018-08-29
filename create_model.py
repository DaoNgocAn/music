from keras.layers import Input, Conv2D, MaxPooling2D, Activation, concatenate, Dropout, \
    Flatten, Dense, BatchNormalization, ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D, Concatenate
from keras.models import Model
from keras import layers, Sequential

import keras.backend as K

def create_model_1(input_shape):
    sq1x1 = "squeeze1x1"
    exp1x1 = "expand1x1"
    exp3x3 = "expand3x3"
    relu = "relu_"

    def fire_module(x, fire_id, squeeze=16, expand=64):
        s_id = 'fire' + str(fire_id) + '/'

        x = Conv2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
        x = Activation('relu', name=s_id + relu + sq1x1)(x)

        left = Conv2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
        left = Activation('relu', name=s_id + relu + exp1x1)(left)

        right = Conv2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
        right = Activation('relu', name=s_id + relu + exp3x3)(right)

        x = concatenate([left, right], axis=3, name=s_id + 'concat')
        return x
    classes = 10

    img_input = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(img_input)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)

    # x = Dropout(0.6, name='drop9')(x)

    x = Conv2D(classes, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax', name='loss')(x)

    return Model(img_input, x, name='squeezenet')

def create_model_2(input_shape):
    classes = 10

    def identity_block(input_tensor, kernel_size, filters, stage, block):
        """The identity block is the block that has no conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names

        # Returns
            Output tensor for the block.
        """
        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size,
                          padding='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)
        return x

    def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        """A block that has a conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            strides: Strides for the first conv layer in the block.

        # Returns
            Output tensor for the block.

        Note that from stage 3,
        the first conv layer at main path is with strides=(2, 2)
        And the shortcut should have strides=(2, 2) as well
        """
        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), strides=strides,
                          name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same',
                          name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                                 name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x

    img_input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    x = Flatten()(x)
    x = Dense(10, activation='softmax')(x)
    return Model(img_input, x, name='resnet50')

def create_model_3(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='resnet'))
    return model

def create_model_4(input_shape):
    classes = 10
    img_input = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    return Model(img_input, x, name='VGG16')

def create_model_5(input_shape):

    def dense_block(x, blocks, name):
        for i in range(blocks):
            x = conv_block(x, growth_rate=32, name=name + '_block' + str(i + 1))
        return x

    def transition_block(x, reduction, name):
        x = BatchNormalization(axis=3, epsilon=1.001e-5,
                               name=name + '_bn')(x)
        x = Activation('relu', name=name + '_relu')(x)
        x = Conv2D(int(K.int_shape(x)[3] * reduction), 1, use_bias=False,
                   name=name + '_conv')(x)
        x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
        return x

    def conv_block(x, growth_rate, name):
        """A building block for a dense block.

        # Arguments
            x: input tensor.
            growth_rate: float, growth rate at dense layers.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        x1 = BatchNormalization(axis=3, epsilon=1.001e-5,
                                name=name + '_0_bn')(x)
        x1 = Activation('relu', name=name + '_0_relu')(x1)
        x1 = Conv2D(4 * growth_rate, 1, use_bias=False,
                    name=name + '_1_conv')(x1)
        x1 = BatchNormalization(axis=3, epsilon=1.001e-5,
                                name=name + '_1_bn')(x1)
        x1 = Activation('relu', name=name + '_1_relu')(x1)
        x1 = Conv2D(growth_rate, 3, padding='same', use_bias=False,
                    name=name + '_2_conv')(x1)
        x = Concatenate(axis=3, name=name + '_concat')([x, x1])
        return x

    def DenseNet(blocks):

        img_input = Input(shape=input_shape)
        x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
        x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
        x = BatchNormalization(axis=3, epsilon=1.001e-5,
                               name='conv1/bn')(x)
        x = Activation('relu', name='conv1/relu')(x)
        x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = MaxPooling2D(3, strides=2, name='pool1')(x)

        x = dense_block(x, blocks[0], name='conv2')
        x = transition_block(x, 0.5, name='pool2')
        x = dense_block(x, blocks[1], name='conv3')
        x = transition_block(x, 0.5, name='pool3')
        x = dense_block(x, blocks[2], name='conv4')
        x = transition_block(x, 0.5, name='pool4')
        x = dense_block(x, blocks[3], name='conv5')

        x = BatchNormalization(axis=3, epsilon=1.001e-5,
                               name='bn')(x)


        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(10, activation='softmax', name='fc')(x)
        return Model(img_input, x, name='densenet')

    return DenseNet([6, 12, 24, 16])
    # return DenseNet([6, 12, 32, 32])
    # return DenseNet([6, 12, 48, 32])

def create_model(name, input_shape):
    if name == 'SqueezeNet':
        return create_model_1(input_shape)
    if name == 'ResNet50':
        return create_model_2(input_shape)
    if name == 'Conv':
        return create_model_3(input_shape)
    if name == 'Conv_2':
        return create_model_4(input_shape)
    if name == 'DenseNet':
        return create_model_5(input_shape)
    return None

if __name__ == '__main__':
    create_model_5((344, 256, 1)).summary()
