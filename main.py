from scipy.io import loadmat

matrixVar = loadmat("Xtrain.mat")
print(matrixVar["Xtrain"])
