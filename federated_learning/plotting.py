import numpy as np
from matplotlib.pyplot import *

#Specify which data to plot
num_hidden = 16
d = 795*num_hidden + 10
k = d
eta_start = 1
eta_final = 1
steps_per_epoch = 10
T = int(d/2)
lmbda = 1e-3
beta = 1e-3
google_colab = False
if google_colab == True:
    foldername = "./drive/MyDrive/Colab Notebooks/Majority_Vote/results"
else:
    foldername = "./results"
train_acc = True

#Data1
com_method1 = "perfect"
optimizer = "sgd"
beta = 0.20
N = 500
k = d
eta_start = 1
eta_final = 1
if com_method1 == "henrik":
  filename = "T=" + str(T) + "_"
else:
  filename = ""
filename = filename + "trainacc_N=" + str(N) + "_k=" + str(k) + "_d=" + str(d) + "_lmbda=" + str(lmbda) + "_beta=" + str(beta) + "_lrs=" + str(eta_start) + "_lrf=" + str(eta_final) + "_optimizer=" + optimizer + "_comMethod=" + com_method1 + "_steps=" + str(steps_per_epoch) +".txt"
path = foldername + "/" + filename
trainacc1 = np.fromfile(path)
test_path = path.replace("train", "test")
testacc1 = np.fromfile(test_path)
norm_path = path.replace("train", "norm")
norm1 = np.fromfile(norm_path)
print("testacc1[-1] =", testacc1[-1])

#Data2
com_method2 = "henrik"
optimizer = "sgd"
beta = 0.10
N = 1000
k = 10
eta_start = 0.4
eta_final = 0.3
T = int(d/2)
if com_method2 == "henrik":
  filename = "T=" + str(T) + "_"
else:
  filename = ""
filename = filename + "trainacc_N=" + str(N) + "_k=" + str(k) + "_d=" + str(d) + "_lmbda=" + str(lmbda) + "_beta=" + str(beta) + "_lrs=" + str(eta_start) + "_lrf=" + str(eta_final) + "_optimizer=" + optimizer + "_comMethod=" + com_method2 + "_steps=" + str(steps_per_epoch) +".txt"
path = foldername + "/" + filename
trainacc2 = np.fromfile(path)
test_path = path.replace("train", "test")
testacc2 = np.fromfile(test_path)
norm_path = path.replace("train", "norm")
norm2 = np.fromfile(norm_path)

#Data3
com_method3 = "perfect_sparse"
optimizer = "sgd"
beta = 0.1
N = 1000
k = 10
eta_start = 1
eta_final = 1
T = int(d/2)
if com_method3 == "henrik":
  filename = "T=" + str(T) + "_"
else:
  filename = ""
filename = filename + "trainacc_N=" + str(N) + "_k=" + str(k) + "_d=" + str(d) + "_lmbda=" + str(lmbda) + "_beta=" + str(beta) + "_lrs=" + str(eta_start) + "_lrf=" + str(eta_final) + "_optimizer=" + optimizer + "_comMethod=" + com_method3 + "_steps=" + str(steps_per_epoch) +".txt"
path = foldername + "/" + filename
trainacc3 = np.fromfile(path)
test_path = path.replace("train", "test")
testacc3 = np.fromfile(test_path)
norm_path = path.replace("train", "norm")
norm3 = np.fromfile(norm_path)

#Data4
com_method4 = "alphan"
optimizer = "sgd"
beta = 0.2
N = 500
k = d
eta_start = 0.2
eta_final = 0.01
t = 0
if com_method4 == "henrik":
  filename = "T=" + str(T) + "_"
elif com_method4 == "alphan":
    filename = "t=" + str(t) + "_"
else:
  filename = ""
filename = filename + "trainacc_N=" + str(N) + "_k=" + str(k) + "_d=" + str(d) + "_lmbda=" + str(lmbda) + "_beta=" + str(beta) + "_lrs=" + str(eta_start) + "_lrf=" + str(eta_final) + "_optimizer=" + optimizer + "_comMethod=" + com_method4 + "_steps=" + str(steps_per_epoch) +".txt"
path = foldername + "/" + filename
trainacc4 = np.fromfile(path)
test_path = path.replace("train", "test")
testacc4 = np.fromfile(test_path)

#-------------- Plotting Norm -------------------------
x1 = np.arange(len(trainacc1))
x2 = np.arange(len(trainacc2))
x3 = np.arange(len(trainacc3))
x4 = np.arange(len(trainacc4))

maxlen = max(len(x1), len(x2), len(x3), len(x4))
x = np.arange(maxlen)

N = len(norm3)
eta_start = 0.01
eta_final = norm3[-1]
normfit3 = eta_start + (eta_final - eta_start)*(1-np.exp(-10*x3/N))

N = len(norm1)
eta_start = 0.1
eta_final = 0.005
normfit1 = eta_start + (eta_final - eta_start)*(1-np.exp(-4*x1/N))

#time = len(x3)
#dec = np.power(norm3[time-1]/norm3[0], 1/time)
#normfit = norm3[0]*np.power(dec, x3[0:time])

#plot(x, testacc1, label=com_method1+" adam k=d")
plot(x1, norm1, label=com_method1)
plot(x3, norm3, label=com_method3+" k=5")
plot(x1, normfit1, label=com_method1+" fit")
plot(x3, normfit3, label=com_method3+" fit")
#plot(x3, testacc3, label=com_method3+" sgd k=d normalized")
#plot(x4, testacc4, label=com_method4+" unnormalized")
legend()
ylim(0.0, 0.8)

#---------------- Extend short -------------------------
cut = 499
if len(trainacc1) < maxlen:
    temp = np.zeros(1000)
    temp[0:cut] = trainacc1[0:cut]
    temp[cut:maxlen] = trainacc1[cut]
    trainacc1 = temp
if len(trainacc2) < maxlen:
    temp = np.zeros(1000)
    temp[0:500] = trainacc2
    temp[500:maxlen] = trainacc2[-1]
    trainacc2 = temp
if len(testacc3) < maxlen:
    temp = np.zeros(1000)
    temp[0:500] = testacc3
    temp[500:maxlen] = testacc3[-1]
    testacc3 = temp
if len(trainacc4) < maxlen:
    temp = np.zeros(1000)
    temp[0:cut] = trainacc4[0:cut]
    temp[cut:maxlen] = trainacc4[cut]
    trainacc4 = temp
    
#---------------- Extend short -------------------------
if len(testacc1) < maxlen:
    temp = np.zeros(1000)
    temp[0:cut] = testacc1[0:cut]
    temp[cut:maxlen] = testacc1[cut]
    testacc1 = temp
if len(testacc2) < maxlen:
    temp = np.zeros(1000)
    temp[0:500] = testacc2
    temp[500:maxlen] = testacc2[-1]
    testacc2 = temp
if len(testacc3) < maxlen:
    temp = np.zeros(1000)
    temp[0:500] = testacc3
    temp[500:maxlen] = testacc3[-1]
    testacc3 = temp
if len(testacc4) < maxlen:
    temp = np.zeros(1000)
    temp[0:cut] = testacc4[0:cut]
    temp[cut:maxlen] = testacc4[cut]
    testacc4 = temp

#-------------- Plotting Train -------------------------
figure()
plot(x, trainacc1, label=com_method1)
plot(x, trainacc2, label=com_method2)
plot(x, trainacc3, label=com_method3)
plot(x, trainacc4, label=com_method4)

legend()
ylim(0.80, 0.93)

#-------------- Plotting Test -------------------------
figure()
plot(x, testacc1, label=com_method1)
plot(x, testacc2, label="our method"+" k=5")
plot(x, testacc3, label=com_method3+" k=5")
plot(x, testacc4, label=com_method4)
legend()
ylim(0.8, 0.96)

print(com_method1, testacc1[-1])
print(com_method2, testacc2[-1])
print(com_method3, testacc3[-1])
print(com_method4, testacc4[-1])
