#If the code is running on Google Colab, I need to mount the GDrive into the workspace to import my homemade dependencies
google_colab = False
if google_colab:
    #Mount google drive into google colab working space
    from google.colab import drive
    drive.mount('/content/drive',force_remount=True)
    #Copy my functions into Colab workspace
    #!cp "/content/drive/MyDrive/Colab Notebooks/Majority_Vote/models_mnist.py" .
    #!cp "/content/drive/MyDrive/Colab Notebooks/Majority_Vote/communication.py" .

#Imports
import models_mnist as models
#import models_cifar as models
import numpy as np
import time
import communication
import tensorflow as tf
import math

#Simulation Variables
n = 10
local_epochs = 1
steps_per_epoch = 1 #None = iterate until entire dataset is covered once
bs = 2**9 #batch size
N = 500 #Number of communication rounds
num_it = N
print(N)
num_hidden = 16;
lmbda = 1e-3;  #l2-Regularization parameter
normalize_update = True
batch_norm = True
dropout = False
error_acc = False
com_norm = False

# Assumes one hidden layer, 10 output neurons, and 784 input neurons.
d = 795*num_hidden + 10;
k = d
if k < d/n:
    r = math.floor(k*n)
    r = 2*k
else:
    r = k    
t = 0
T = int(d)
round_list = np.arange(int(d/T*N))
c_star = 0.40 #optimal for k=1
d1 = 1
d2 = 100
# Learning rate decays every decay/freq/(T/d) communication rounds
eta_start = 0.1
eta_final = 0.005
norm_sch = eta_start + (eta_final - eta_start)*(1-np.exp(-10*round_list/N))
beta = 0.20
optimizer = "sgd"
decay_freq = 1
#com_methods = ["perfect", "perfect_sparse", "alphan", "henrik"]
com_methods = ["alphan"]

#Download global dataset and split it into K local datasets
local_datasets, global_dataset = models.load_dataset(n)
#CIFAR labels
#labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
#MNIST labels
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
#Create communication model
com = communication.CommunicationModel(sigma_z=0.1, sigma_h=1, P=1, R=2, beta=2, d=d, delta=3, c=c_star)

for com_method in com_methods:
    #Number of iterations depend on cost of running the system
    #Fix seeds for fair comparison
    np.random.seed(123)
    tf.random.set_seed(123)
    print("com =", com_method)
    #Reset LR
    lr = eta_start

    #Create n local models and one global model with randomly initialized weights
    model_list, global_model = models.setup_models(n, len(labels), num_hidden, lmbda, beta, d1, d2, optimizer)

    #History vectors for plotting
    train_acc = np.zeros((num_it, 1))
    test_acc = np.zeros((num_it, 1))
    weight_norm = np.zeros((num_it, 1))
    mean_norms = np.zeros((num_it, 1))

    #Initialize error accumulation for sparse GD
    delta = np.zeros((d, n))

    #Federated learning loop
    sch_idx = 0
    start = time.time()
    for i in range(num_it):
        print("Communication round " + str(i+1) + ".")
        #Learning rate schedule
        #if (i)%(sched_update) == 0 and i > 0:
        #    lr = lr*lr_decay
        #Learning rate schedule
        lr = norm_sch[i]
        if (i)%(10) == 0 and i > 0:
            print("train_acc[i-1]", train_acc[i-1])
            print("test_acc[i-1]", test_acc[i-1])
            print("lr =", lr)
            #print("model_list[0].optimizer.learning_rate.numpy() =", model_list[0].optimizer.learning_rate.numpy())
            
        if error_acc == False:
            delta = np.zeros((d, n))
        #Broadcast to devices
        model_list, global_model = models.model_broadcast(model_list, global_model)

        #Local training
        model_list = models.local_training(model_list, local_datasets, local_epochs, steps_per_epoch, bs)

        #Get model deltas
        model_updates = models.get_update(model_list, global_model)

        #Sparsify
        if (com_method == "henrik") or (com_method == "perfect_sparse"):
            update_matrix, delta, mean_norm = models.sparsifyRTopKRandom(model_updates, delta, d, k, r)
        elif (com_method == "perfect"):
            #update_matrix, delta, mean_norm = models.sparsifyTopK(model_updates, delta, d, d)
            update_matrix, delta, mean_norm = models.sparsifyAlphan(model_updates, delta, d, t)
        else:
            update_matrix, delta, mean_norm = models.sparsifyAlphan(model_updates, delta, d, t)
            
        nnz = np.count_nonzero(update_matrix[:, 1])
        print("nnz =", nnz)
            
        #Communication & Aggregation
        if (com_method == "perfect") or (com_method == "perfect_sparse"):
            update_vector = com.perfect(update_matrix)
        elif com_method == "alphan":
            update_vector = com.alphan(update_matrix)
        elif com_method == "henrik":
            update_vector = com.henrik(update_matrix, T)
        else:
            print("ERROR: Cannot recognize the chosen communication method: ", com_method)
            break
        
        #print("np.count_nonzero(update_vector) =", np.count_nonzero(update_vector))
        
        if com_norm == True:
            lr = mean_norm
        
        weight_norm[i] = np.linalg.norm(update_vector)
        mean_norms[i] = mean_norm
        
        #Global model update step (inside AP CPU)
        global_model = models.global_update(update_vector, global_model, lr, normalize_update)

        #Store history
        train_acc[i] = global_model.evaluate(global_dataset[0], global_dataset[1], batch_size=None, verbose=0)[1]
        test_acc[i] = global_model.evaluate(global_dataset[2], global_dataset[3], batch_size=None, verbose=0)[1]
        
        

    end = time.time()
    print("Time spent on Federated Learning =", end-start)

    #Accuracy evaluation
    results = global_model.evaluate(global_dataset[0], global_dataset[1], batch_size=128, verbose=0)
    print("train loss, train acc of global model:", results)
    results = global_model.evaluate(global_dataset[2], global_dataset[3], batch_size=128, verbose=0)
    print("test loss, test acc of global model:", results)

    #Interpolate such that all accuracy vectors have equal length
    temp = np.zeros(len(train_acc))
    temp2 = np.zeros(len(train_acc))
    for i in range(len(train_acc)):
        temp[i] = train_acc[i][0]
        temp2[i] = test_acc[i][0]
    train_acc = temp
    test_acc = temp2
    #if relative_cost == 1:
    #    x = np.linspace(1, num_it, num_it)
    #    relative_cost = T/d
    #    x_interp = np.linspace(1,num_it,int(num_it/relative_cost))
    #    train_acc = np.interp(x_interp, x, train_acc)
    #    test_acc = np.interp(x_interp, x, test_acc)

    #Store results
    if google_colab == True:
        foldername = "./drive/MyDrive/Colab Notebooks/Majority_Vote/results"
    else:
        foldername = "./results"

    if com_method == "henrik":
        filename = "T=" + str(T) + "_"
    elif com_method == "alphan":
        filename = "t=" + str(t) + "_"
    else:
        filename = ""
    filename = filename + "trainacc_N=" + str(N) + "_k=" + str(k) + "_r=" + str(r) + "_d="+ str(d) + "_lmbda=" + str(lmbda) + "_beta=" + str(beta) + "_lrs=" + str(eta_start) + "_lrf=" + str(eta_final) + "_optimizer=" + optimizer + "_comMethod=" + com_method + "_steps=" + str(steps_per_epoch) +".txt"
    path = foldername + "/" + filename
    train_acc.tofile(path)
    path = path.replace("train", "test")
    test_acc.tofile(path)
    path = path.replace("test", "norm")
    weight_norm.tofile(path)
    print("saving data to", filename)