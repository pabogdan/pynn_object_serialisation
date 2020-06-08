import argparse
import tensorflow as tf
import tensorflow.keras as keras
#import keras
from keras_rewiring.sparse_layer import Sparse, SparseConv2D, SparseDepthwiseConv2D
import shap
import numpy as np
np.random.seed(42)

c_obj = {'Sparse': Sparse,
             'SparseConv2D': SparseConv2D,
             'SparseDepthwiseConv2D': SparseDepthwiseConv2D}

parser = argparse.ArgumentParser(
    description='SHAP experiment argparser',
    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('model', type=str,
                    help='network architecture / model to analyse')

args = parser.parse_args()

#Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_test = x_test.reshape((-1,1, 784))
x_train = x_train.reshape((-1,1, 784))
y_test = keras.utils.to_categorical(y_test, 10)
y_train = keras.utils.to_categorical(y_train, 10)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#Load an ANN model
model = keras.models.load_model(args.model, compile=True, custom_objects=c_obj)

def change_batch_size(model, batch_size, c_obj):
    #This is the easiest way of changing batch_size apparently...
    config = model.get_config()

    for layer in config['layers']:
        if 'batch_input_shape' in layer['config']:
            batch_input_shape = list(layer['config']['batch_input_shape'])
            batch_input_shape[0] = batch_size
            layer['config']['batch_input_shape'] = tuple(batch_input_shape)
    new_model = model.from_config(config, custom_objects=c_obj)
    for layer in new_model.layers:
        # copy weights from old model to new one
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
        except:
            print("Could not transfer weights for layer {}".format(layer.name))

    # rebuild model architecture by exporting and importing via json
    #new_model = keras.models.model_from_json(model.to_json(), custom_objects=c_obj)



    return new_model

#model = change_batch_size(model, None, c_obj)


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# model.fit(
#    x_train, y_train,
#    epochs = 20,
#    verbose = 1,
#    validation_data = (x_test, y_test)
# )

model.summary()



score = model.evaluate(x_test, y_test, verbose=0, batch_size=100)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print(np.argmax(model.predict(x_test[0:10]), axis=1))




#Do the SHAP analysis
#Taken from shap github
# select a set of background examples to take an expectation over

background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)].reshape((-1, 1, 784))

# explain predictions of the model on four images

e = shap.DeepExplainer(model, background)

index = 9

shap_values = e.shap_values(x_test[0:10], check_additivity=True)
reshaped_shap_values = [example.reshape((-1, 28, 28)) for example in shap_values]
#Making meaningful labels
labels = np.expand_dims(np.arange(0,10), axis=0)
empty = np.repeat(np.zeros_like(labels, dtype=str),9, axis=0)
empty[:] = ''
labels = np.vstack((labels, empty))

# plot the feature attributions
shap.image_plot(reshaped_shap_values, np.array(-x_test[0:10]).reshape((-1, 28, 28)), labels=labels)