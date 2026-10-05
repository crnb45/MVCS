import numpy as np
from matplotlib.pyplot import *

#Specify which data to plot
num_hidden = 16
d = 795*num_hidden + 10
k = d
eta_start = 1
eta_final = 1
steps_per_epoch = 1
optimizer = "sgd"
T = int(d/2)
lmbda = 1e-3
beta = 1e-3
com_method = "perfect_sparse"
k = 5
N = 1000
google_colab = False
if google_colab == True:
    foldername = "./drive/MyDrive/Colab Notebooks/Majority_Vote/results"
else:
    foldername = "./results"
train_acc = True

#Data1
optimizer = "sgd"
beta1 = 0.05
eta_start = 1
eta_final = 1
if com_method == "henrik":
  filename = "T=" + str(T) + "_"
else:
  filename = ""
filename = filename + "trainacc_N=" + str(N) + "_k=" + str(k) + "_d=" + str(d) + "_lmbda=" + str(lmbda) + "_beta=" + str(beta1) + "_lrs=" + str(eta_start) + "_lrf=" + str(eta_final) + "_optimizer=" + optimizer + "_comMethod=" + com_method + "_steps=" + str(steps_per_epoch) +".txt"
path = foldername + "/" + filename
trainacc1 = np.fromfile(path)
test_path = path.replace("train", "test")
testacc1 = np.fromfile(test_path)
norm_path = path.replace("train", "norm")
norm1 = np.fromfile(norm_path)
print("testacc1[-1] =", testacc1[-1])

#Data2
optimizer = "sgd"
beta2 = 0.10
eta_start = 1
eta_final = 1
if com_method == "henrik":
  filename = "T=" + str(T) + "_"
else:
  filename = ""
filename = filename + "trainacc_N=" + str(N) + "_k=" + str(k) + "_d=" + str(d) + "_lmbda=" + str(lmbda) + "_beta=" + str(beta2) + "_lrs=" + str(eta_start) + "_lrf=" + str(eta_final) + "_optimizer=" + optimizer + "_comMethod=" + com_method + "_steps=" + str(steps_per_epoch) +".txt"
path = foldername + "/" + filename
trainacc2 = np.fromfile(path)
test_path = path.replace("train", "test")
testacc2 = np.fromfile(test_path)
norm_path = path.replace("train", "norm")
norm2 = np.fromfile(norm_path)

#Data3
optimizer = "sgd"
beta3 = 0.20
eta_start = 1
eta_final = 1
if com_method == "henrik":
  filename = "T=" + str(T) + "_"
else:
  filename = ""
filename = filename + "trainacc_N=" + str(N) + "_k=" + str(k) + "_d=" + str(d) + "_lmbda=" + str(lmbda) + "_beta=" + str(beta3) + "_lrs=" + str(eta_start) + "_lrf=" + str(eta_final) + "_optimizer=" + optimizer + "_comMethod=" + com_method + "_steps=" + str(steps_per_epoch) +".txt"
path = foldername + "/" + filename
trainacc3 = np.fromfile(path)
test_path = path.replace("train", "test")
testacc3 = np.fromfile(test_path)
norm_path = path.replace("train", "norm")
norm3 = np.fromfile(norm_path)

#Data4
optimizer = "sgd"
beta4 = 0.20
eta_start = 1
eta_final = 1
if com_method == "henrik":
  filename = "T=" + str(T) + "_"
elif com_method == "alphan":
    filename = "t=" + str(t) + "_"
else:
  filename = ""
filename = filename + "trainacc_N=" + str(N) + "_k=" + str(k) + "_d=" + str(d) + "_lmbda=" + str(lmbda) + "_beta=" + str(beta4) + "_lrs=" + str(eta_start) + "_lrf=" + str(eta_final) + "_optimizer=" + optimizer + "_comMethod=" + com_method + "_steps=" + str(steps_per_epoch) +".txt"
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
eta_start = 0.2
eta_final = norm3[-1]
normfit3 = eta_start + (eta_final - eta_start)*(1-np.exp(-4*x3/N))

N = len(norm1)
eta_start = 0.1
eta_final = 0.005
normfit1 = eta_start + (eta_final - eta_start)*(1-np.exp(-4*x1/N))

#time = len(x3)
#dec = np.power(norm3[time-1]/norm3[0], 1/time)
#normfit = norm3[0]*np.power(dec, x3[0:time])

#plot(x, testacc1, label=com_method+" adam k=d")
plot(x1, norm1, label="beta="+str(beta1))
plot(x3, norm3, label="beta="+str(beta3))
plot(x1, normfit1, label="fit beta="+str(beta1))
plot(x3, normfit3, label="fit beta="+str(beta3))
#plot(x3, testacc3, label=com_method+" sgd k=d normalized")
#plot(x4, testacc4, label=com_method+" unnormalized")
legend()
ylim(0.0, 0.25)

#---------------- Extend short -------------------------
if len(testacc1) < maxlen:
    temp = np.zeros(1000)
    temp[0:500] = testacc1
    temp[500:maxlen] = testacc1[-1]
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
    temp[0:500] = testacc4
    temp[500:maxlen] = testacc4[-1]
    testacc4 = temp

#-------------- Plotting Train -------------------------
figure()
plot(x1, trainacc1, label="beta="+str(beta1))
plot(x2, trainacc2, label="beta="+str(beta2))
plot(x3, trainacc3, label="beta="+str(beta3))
plot(x4, trainacc4, label="beta="+str(beta4))

legend()
ylim(0.50, 0.962)

#-------------- Plotting Test -------------------------
figure()
plot(x, testacc1, label="beta="+str(beta1))
plot(x, testacc2, label="beta="+str(beta2))
plot(x, testacc3, label="beta="+str(beta3))
plot(x, testacc4, label="beta="+str(beta4))
legend()
ylim(0.85, 0.93)

print(com_method, testacc1[-1])
print(com_method, testacc2[-1])
print(com_method, testacc3[-1])
print(com_method, testacc4[-1])
