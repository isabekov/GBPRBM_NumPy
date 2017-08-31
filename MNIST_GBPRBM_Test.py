import numpy as np
from scipy.io import savemat,loadmat
from GBPRBM import GBPRBM
from MNIST_Plot_Image_Pairs import MNIST_Plot_Image_Pairs

# Initialize a GBPRBM class instance with previously saved model parameters
PL = loadmat("Model,V=784,H=500.mat")
GBPRBM_Model = GBPRBM(PL=PL)

DataFile = "MNIST_Test_Medal_Normalized.mat"
MNIST_Test = loadmat(DataFile)

# Number of samples in the test data set
N_Samples_Test = MNIST_Test["testLabels"].shape[0]
# Number of images per digit to display
N_IPD = 12
# Number of unique digits
N_UD = 10
idx = np.zeros([N_IPD, N_UD], dtype=int)
Num_Idx = np.arange(N_UD)
for j in range(N_UD):
    cnt = 0
    for k in range(N_Samples_Test):
        # Boolean to numberical indexing
        Digit = Num_Idx[MNIST_Test["testLabels"][k,:].astype(bool)]
        if Digit == j:
            idx[cnt,j] = k
            cnt = cnt + 1
        if cnt == N_IPD:
           break
idx = idx.reshape(N_IPD*N_UD)
# DataTrain: N_Pixels x N_Samples
DataTrain = MNIST_Test["testData"]

Img_Orig = DataTrain[idx,:].transpose()
# Reconstruct images using GBPRBM
[_, Img_Recon] = GBPRBM_Model.Test(Img_Orig)
# Plot image pairs
MNIST_Plot_Image_Pairs(Img_Orig, Img_Recon, N_Horiz = N_IPD, N_Vert = N_UD, str = "MNIST Original and Reconstructed Images")