# -*- coding: utf-8 -*-
"""
Script used to define the comparisons model and trained it.
"""
# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import os
import numpy as np

from keras import Input, Model, Sequential
from keras.applications import VGG19, ResNet152V2, ResNet50V2, VGG16, Xception
from keras.layers import concatenate, Conv2D, Dropout, Flatten, Dense, BatchNormalization, AveragePooling2D, Reshape, Activation, Subtract, Concatenate
from keras.optimizers import SGD, Adam


from keras.layers.experimental.preprocessing import RandomTranslation, RandomFlip, RandomRotation, RandomZoom, RandomHeight, RandomWidth,Rescaling 

from keras.applications.imagenet_utils import preprocess_input

import tensorflow as tf

# ----------------------------------------------------------------------------------------------------------------------
# Functions definitions
# ----------------------------------------------------------------------------------------------------------------------

def map_fn(serialized_example):
    feature = {
        'data_label': tf.io.FixedLenFeature([2], tf.int64),
        'data_right': tf.io.FixedLenFeature([150528], tf.int64),
        'data_left': tf.io.FixedLenFeature([150528], tf.int64),
    }
    
    ex = tf.io.parse_single_example(serialized_example, feature)
    #ex['data_label'] = ex['data_label'][0]
    return ex, ex['data_label']

def preprocessing_layers(a, hp):
    #a = preprocess_input(a)#Rescaling(1/255)(a)
    translate = hp.Float("translate", 0, 0.5, step=0.1, default=0.2)
    a = RandomTranslation( 
        #hp.Float("t_x", 0, 0.5, step=0.1, default=0.2), 
        #hp.Float("t_y", 0, 0.5, step=0.1, default=0.2), 
        translate,
        translate,
        fill_mode="reflect",interpolation="bilinear",)(a)
    a = RandomFlip()(a)
    a = RandomZoom(hp.Float("zoom", 0, 0.5, step=0.1, default=0.25))(a)
    a = RandomRotation(hp.Float("rotation", 0, 2, step=0.1, default=2))(a)
    #a = RandomHeight(0.2)(a)
    #a = RandomWidth(0.2)(a)
    
    return a

data_augmentation = Sequential([
                                RandomFlip("horizontal_and_vertical"),
                                RandomRotation((0,0.5), fill_mode='constant')
])
def comparisons_model(img_size, weigths=None):
    """
    Create comparisons network which reproduce the choice in an images duel.
    :param img_size: size of input images during training
    :type img_size: tuple(int)
    :param weights: path to the weights use for initialization
    :type weights: str
    :return: ranking comparisons model
    :rtype: keras.Model
    """
    vgg_feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

    # Fine tuning by freezing the last 4 convolutional layers of VGG19 (last block)
    for layer in vgg_feature_extractor.layers[:-4]:
        layer.trainable = False

    # Definition of the 2 inputs
    img_a = Input(shape=(224*224*3), name="data_left")
    out_a = Reshape((224,224,3), input_shape=(224*224*3,))(img_a)
    
    img_b = Input(shape=(224*224*3), name="data_right")
    out_b = Reshape((224,224,3), input_shape=(224*224*3,))(img_b)
    
    out_a = data_augmentation(out_a)
    out_b = data_augmentation(out_b)

    out_a = vgg_feature_extractor(out_a)
    out_b = vgg_feature_extractor(out_b)

    # Concatenation of the inputs
    concat = concatenate([out_a, out_b])

    # Add convolution layers on top
    x = Conv2D(1024, (3, 3), padding='same', name="Conv_1")(concat)
    x = BatchNormalization()(x)
    x = Activation('relu', name='Activation_1')(x)
    x = Dropout(0.66, name="Drop_1")(x)
    x = Conv2D(512, (3, 3), padding='same', name="Conv_2")(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='Activation_2')(x)
    x = Conv2D(256, (3, 3), padding='same', name="Conv_3")(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='Activation_3')(x)
    x = Conv2D(128, (3, 3), padding='same', name="Conv_4")(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='Activation_4')(x)
    x = Dropout(0.5, name="Drop_2")(x)
    x = Flatten()(x)
    x = Dense(2, activation='softmax', name="Final_dense")(x)

    classification_model = Model([img_a, img_b], x)
    if weigths:
        classification_model.load_weights(weigths)
    #sgd = SGD(learning_rate=1e-5, decay=1e-6, momentum=0.3, nesterov=True)
    adam = Adam(learning_rate=1e-5)
    classification_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return classification_model

def ranking_model(img_size, vgg_feature_extractor=None, weights=None):
    img_size=224
    weights=None
    """
    Create comparisons network which reproduce the choice in an images duel.

    :param img_size: size of input images during training
    :type img_size: tuple(int)
    :param weights: path to the weights use for initialization
    :type weights: str
    :return: ranking comparisons model
    :rtype: keras.Model
    """
    
    #vgg_feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    
    #vgg_include_until='block4_pool'
    #feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    #feature_extractor = Model(inputs=vgg_feature_extractor.input, outputs=feature_extractor.get_layer(vgg_include_until).output)
    if vgg_feature_extractor is None:
        vgg_feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

    # Fine tuning by freezing the last 4 convolutional layers of VGG19 (last block)
    #for layer in vgg_feature_extractor.layers[:-hp.Int(
    #    'layers_frozen',
    #    min_value=0,
    #    max_value=22,
    #    step=2,
    #   default=4
    #)]:
    #    layer.trainable = False

    # Definition of the 2 inputs
    img_a = Input(shape=(224*224*3), name="data_left")
    out_a = Reshape((224,224,3), input_shape=(224*224*3,))(img_a)
    
    img_b = Input(shape=(224*224*3), name="data_right")
    out_b = Reshape((224,224,3), input_shape=(224*224*3,))(img_b)
    
    out_a = data_augmentation(out_a)
    out_b = data_augmentation(out_b)

    out_a = vgg_feature_extractor(out_a)
    out_b = vgg_feature_extractor(out_b)

    # Concatenation of the inputs
    
    def ranking_layers(x, name):
        x = Flatten()(x)
        x = Dense(4096, name="Dense_1_{}".format(name))(x)
        x = BatchNormalization()(x)
        x = Activation('relu', name='Activation_1_{}'.format(name))(x)
        x = Dropout(0.5, name="Drop_1_{}".format(name))(x)
        
        x = Dense(4096, name="Dense_2_{}".format(name))(x)
        x = BatchNormalization()(x)
        x = Activation('relu', name='Activation_2_{}'.format(name))(x)
        x = Dropout(0.5, name="Drop_2_{}".format(name))(x)
        
        x = Dense(1, name="Dense_3_{}".format(name))(x)
        return x
    
    out_a = ranking_layers(out_a, "a")
    out_b = ranking_layers(out_b, "b")
    #concat = concatenate([out_a, out_b])
    x = Subtract()([out_a, out_b])
    x = Activation('sigmoid', name='Activation_2')(x)
    
    
    #x=Concatenate(axis=-1)[x, Subtract()([1, x])]
    x=Concatenate(axis=-1)([x, Subtract()([np.array([1.0]), x])])
    classification_model = Model([img_a, img_b], x)
    #if weigths is not None:
    #    classification_model.load_weights(weigths)
    #sgd = SGD(learning_rate=1e-5, decay=1e-6, momentum=0.3, nesterov=True)
    classification_model.compile(loss='categorical_crossentropy', 
                                 optimizer=Adam(5e-5), 
                                 metrics=['accuracy'])
    return classification_model
