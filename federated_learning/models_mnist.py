#Dependencies
import tensorflow as tf
from tensorflow import keras
import time
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import utils
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np

#Global variables
layers_with_weights = ["dense"]

#n = number of devices
#local_datasets = list with n entries where each entry is a tuple with 2 entires: train_image, train_label
#global_dataset = tuple with 4 entires: train_image, train_label, test_image, test_label
def load_dataset(n):
    #Dataset
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # One hot encoding the target class (labels)
    num_classes = 10
    train_labels = utils.to_categorical(train_labels, num_classes)
    test_labels = utils.to_categorical(test_labels, num_classes)
    global_dataset = (train_images, train_labels, test_images, test_labels)
    
    #Split global dataset into local datasets for each device
    local_datasets = __split_dataset(n, global_dataset)
    return local_datasets, global_dataset

def __split_dataset(n, dataset):
    N_train = dataset[0].shape[0]
    ret = []
    
    for k in range(n):
        #Train data
        start = k*int(N_train/n)
        end = int(N_train/n) + k*int(N_train/n)
        train_x_k = dataset[0][start:end]
        train_y_k = dataset[1][start:end]
        ret.append((train_x_k, train_y_k))
        
    return ret

#Model setup
def setup_models(n, num_classes, num_hidden, lmbda, beta, d1, d2, optimizer):
    model_list = []
    for k in range(n):
        model_list.append(__initialize_model(num_classes, num_hidden, lmbda, beta, d1, d2, optimizer))
    global_model = __initialize_model(num_classes, num_hidden, lmbda, beta, d1, d2, optimizer)
    return model_list, global_model

def __initialize_model(num_classes, num_hidden, lmbda, beta, d1, d2, optimizer):
    #The weights of the layers are initialized when they are added to the Sequential object
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(num_hidden, kernel_regularizer=regularizers.L2(lmbda), activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=regularizers.L2(lmbda))
    ])
    
    #Create optimizer
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=beta,
    decay_steps=d2,
    decay_rate=d1,
    staircase=True)
    opt = SGD(learning_rate=lr_schedule, momentum=0)
    if optimizer == "adam":
        opt = tf.keras.optimizers.Adam(beta)
    #opt.learning_rate.assign(beta)
    #opt.beta_1 = d1
    #opt.beta_2 = d2
    #Loss
    loss_fn = tf.keras.losses.categorical_crossentropy
    
    model.compile(optimizer=opt, loss=loss_fn, metrics='accuracy')
    return model

#Trains all models in model_list
def local_training(model_list, mnist_datasets, ep, steps, bs):
    #Fix seeds for fair comparison
    for k, model in enumerate(model_list):
        train_images = mnist_datasets[k][0]
        train_labels = mnist_datasets[k][1]
        history = model.fit(train_images, train_labels, batch_size=bs, epochs=ep, steps_per_epoch=steps, verbose=0)
    return model_list

#Takes the trained models and the global model as input and returns the diff
def get_update(model_list, global_model):
    num_layers = len(global_model.layers)

    for l in range(num_layers):
        layer_name = global_model.layers[l].name.split("_")[0]
        if layer_name in layers_with_weights:
            weight_list_len = len(global_model.layers[l].weights)
            for w_idx in range(weight_list_len):
                delta = \
                    tf.zeros(global_model.layers[l].weights[w_idx].get_shape())
                global_weights = global_model.layers[l].weights[w_idx]
                for model in model_list:
                    local_weights = model.layers[l].weights[w_idx]
                    #delta = w_g - w_i
                    delta = global_weights - local_weights
                    #Set weights of the local model to the delta
                    model.layers[l].weights[w_idx].assign(delta)
    return model_list
        
#Takes a list of model updates as input, returns d by n matrix where each
#column is sparsified down to top-k in magnitude and accumulated error
def sparsifyTopK(model_updates, delta, d, k):
    n = len(model_updates)
    ret = np.zeros((d, n))
    norm_sum = 0
    for i, update in enumerate(model_updates):
        weight_vec = __weights_to_vector(update, d)
        norm_sum = norm_sum + np.linalg.norm(weight_vec)
        #Gradient + accumulated error
        params = weight_vec + delta[:, i]
        #Get top-k elements
        idxs = np.flip( np.argsort(np.abs(params)) )
        sparse_params = params.copy()
        sparse_params[idxs[k:]] = 0
        #Update accumulated error
        delta[:, i] = params - sparse_params
        #Store in ret matrix
        ret[:,i] = sparse_params
    mean_norm = norm_sum/n
    return ret, delta, mean_norm
              
#Takes a list of model updates as input, returns d by n matrix where each
#column is sparsified down to top-r in magnitude and then random top-k out of 
#those top-r
def sparsifyRTopKRandom(model_updates, delta, d, k, r):
    n = len(model_updates)
    ret = np.zeros((d, n))
    norm_sum = 0
    for i, update in enumerate(model_updates):
        weight_vec = __weights_to_vector(update, d)
        norm_sum = norm_sum + np.linalg.norm(weight_vec)
        #Gradient + accumulated error
        params = weight_vec + delta[:, i]
        #Get top-k elements
        idxs = np.flip( np.argsort(np.abs(params)) )
        toprk = idxs[0:r]
        topk = np.random.permutation(toprk)
        topk = topk[0:k]
        sparse_params = np.zeros(params.shape)
        sparse_params[topk] = params[topk]
        #Update accumulated error
        delta[:, i] = params - sparse_params
        #Store in ret matrix
        ret[:,i] = sparse_params
    mean_norm = norm_sum/n
    return ret, delta, mean_norm
 
#Takes a list of model updates as input, returns d by n matrix where each
#column is sparsified such that entries of absolute value below t are removed.
def sparsifyAlphan(model_updates, delta, d, t):
    n = len(model_updates)
    ret = np.zeros((d, n))
    norm_sum = 0
    for i, update in enumerate(model_updates):
        weight_vec = __weights_to_vector(update, d)
        norm_sum = norm_sum + np.linalg.norm(weight_vec)
        #Gradient + accumulated error
        params = weight_vec + delta[:, i]
        #Deep copy params for delta calculation later
        sparse_params = params.copy()
        #Only keep elements with greater absolute value than t
        keep = np.greater(np.abs(params), t)
        sparse_params = np.multiply(sparse_params, keep)
        #Update accumulated error
        delta[:, i] = params - sparse_params
        #Store in ret matrix
        ret[:,i] = sparse_params
    mean_norm = norm_sum/n
    return ret, delta, mean_norm

#Takes model as input and returns a vector of all parameters in that model
#d = number of params in model (for preallocation)
def __weights_to_vector(model, d):
    num_layers = len(model.layers)
    params = np.zeros(d)
    i = 0 #tracker for params vector
    for l in range(num_layers):
        layer_name = model.layers[l].name.split("_")[0]
        if layer_name in layers_with_weights:
            #See global_update() for explanation to weight_list_len
            weight_list_len = len(model.layers[l].weights)
            for w_idx in range(weight_list_len):
                layer_weights = model.layers[l].weights[w_idx]
                vals = K.get_value(layer_weights).flatten()
                params[i:i+len(vals)] = vals
                i = i+len(vals)
    return params

#Takes vector of parameters as input, returns model with params as weights and
#model structure according to "model" input
def __vector_to_weights(model, params):
    num_layers = len(model.layers)
    i = 0 #tracker for params vector
    for l in range(num_layers):
        layer_name = model.layers[l].name.split("_")[0]
        if layer_name in layers_with_weights:
            #See global_update() for explanation to weight_list_len
            weight_list_len = len(model.layers[l].weights)
            for w_idx in range(weight_list_len):
                layer_weights = model.layers[l].weights[w_idx]
                layer_shape = layer_weights.shape
                #Some layers are vectors and some are matrices
                if len(layer_shape) > 1:
                    layer_len = layer_shape[0]*layer_shape[1]
                else:
                    layer_len = layer_shape[0]
                vals = params[i:i+layer_len]
                vals = vals.reshape(layer_shape)
                #Asign weights back into model
                model.layers[l].weights[w_idx].assign(vals)
                i = i+layer_len
    return model

#Update global_model weights as 
#global_model^(new)=global_model - model_matrix
def global_update(model_vector, global_model, lr, normalize):
    num_layers = len(global_model.layers)
    model = models.clone_model(global_model)
    model = __vector_to_weights(model, model_vector)
    weight_norm = np.linalg.norm(model_vector)
    print("weight_norm =", weight_norm)

    for l in range(num_layers):
        layer_name = global_model.layers[l].name.split("_")[0]
        if layer_name in layers_with_weights:
            #Depending on the layer, there will be different kinds of weights. 
            #Some layers have only weights+biases, in which case len(global_model.layers[l].weights)=2.
            #Some layers also have configurable parameters, such as the BatchNormalization layer
            #which has 2 extra parameters (gamma and beta) leading to len(global_model.layers[l].weights)=4.
            weight_list_len = len(global_model.layers[l].weights)
            for w_idx in range(weight_list_len):
                weight_update = model.layers[l].weights[w_idx]
                
                if normalize == True:
                    if weight_norm == 0:
                        weight_norm = 1
                    weight_update = weight_update/weight_norm
                new_weights = global_model.layers[l].weights[w_idx] - lr*weight_update
                global_model.layers[l].weights[w_idx].assign(new_weights)

    return global_model
#Update the learning rate of all models
def update_lr(model_list, global_model, eta):
    global_model.optimizer.learning_rate.assign(eta)
    for model in model_list:
        model.optimizer.learning_rate.assign(eta)    
                
    return model_list, global_model

#Sets the weights of all models in model_list to equal those of global_model
def model_broadcast(model_list, global_model):
    num_layers = len(global_model.layers)

    for l in range(num_layers):
        layer_name = global_model.layers[l].name.split("_")[0]
        if layer_name in layers_with_weights:
            #See global_update() for explanation to weight_list_len
            weight_list_len = len(global_model.layers[l].weights)
            for w_idx in range(weight_list_len):
                global_weights = global_model.layers[l].weights[w_idx]
                for model in model_list:
                    model.layers[l].weights[w_idx].assign(global_weights)
                
    return model_list, global_model