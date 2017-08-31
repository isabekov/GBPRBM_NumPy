import os.path
from scipy.io import loadmat
from MNIST_Download import MNIST_Download
import GBPRBM

# Initialize a GBPRBM class instance
GBPRBM_Model = GBPRBM.GBPRBM(V=784,H=500, sigma_v=0.0001)

DataFile = "MNIST_Train_Medal_Normalized.mat"
if not os.path.isfile(DataFile):
    MNIST_Download() # Download and convert the MNIST database
MNIST_Train = loadmat(DataFile)
# DataTrain: N_Pixels x N_Samples
DataTrain = MNIST_Train["trainData"].transpose()

# Train a GBPRBM model using data provided
RMSE_Train = GBPRBM_Model.Train(Data = DataTrain, nu=1e-10, mu=0.8, mBS = 50, N_Epochs=1)
# Save model parameters
GBPRBM_Model.SaveModel()