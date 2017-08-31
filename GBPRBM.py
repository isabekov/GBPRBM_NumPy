import os
import numpy as np
import guiqwt.pyplot as plt
from scipy.io import savemat

class GBPRBM(object):
    def __init__(self,
                 V = None,  # Number of visible units
                 H = None,  # Number of hidden units
                 sigma_v = None,
                 PL = None):  # Parameters loaded

        if isinstance(PL, dict):
            Keys = ["W", "b_v", "b_h", "sigma_v"]
            for key in Keys:
                if key in PL:
                    setattr(self, key, PL[key])
            if (V is None) and ("W" in PL):
                self.V = PL["W"].shape[0]
            if (H is None) and ("W" in PL):
                self.H = PL["W"].shape[1]
        else:
            self.V = V
            self.H = H
            self.W   = 0.01 * np.random.random([self.V, self.H])
            self.b_h = 0.01 * np.random.random([self.H, 1])
            if isinstance(sigma_v, float) or isinstance(sigma_v, int):
                self.sigma_v = sigma_v*np.ones([self.V,1])
            elif isinstance(sigma_v, np.ndarray):
                if sigma_v.shape == (self.V,1):
                    self.sigma_v = sigma_v
            else:
                self.sigma_v = 0.001 * np.ones([self.V, 1])
        self.sv_sq = self.sigma_v ** 2
        self.sv_sq_ts = np.expand_dims(self.sv_sq, axis=1)

    def Train(self,
              Data,
              mu = 0.5,       # momentum
              nu = 1e-7,      # learning rate
              mBS=100,
              N_Epochs=3):
        if not 0.0 <= mu <= 1.0:
            raise ValueError('Momentum should be in range [0, 1]')
        self.mu = mu
        self.nu = nu
        display_step = 1
        if not hasattr(self, "b_v"):
            self.b_v = Data.mean(axis=1, keepdims=True)

        # Differentials
        self.D_W   = np.zeros([self.V, self.H])
        self.D_b_v = np.zeros([self.V, 1])
        self.D_b_h = np.zeros([self.H, 1])

        N_Samples_Train = Data.shape[1]
        # Number of mini-batches
        N_mBatches = int(np.ceil(N_Samples_Train / mBS))
        Digits = np.log10(N_mBatches)
        if Digits.is_integer():
            N_Digs = int(Digits) + 1
        else:
            N_Digs = int(np.ceil(Digits))
        # Memory preallocation for training RMSE
        RMSE_Train = np.zeros(N_mBatches*N_Epochs)
        print("Starting training")
        for Epoch in range(N_Epochs):
            print("=============== Epoch {:d} ===============".format(Epoch + 1))
            # Sample indices for shuffling data
            idx = np.random.permutation(N_Samples_Train)
            # Loop over all batches
            for i in range(N_mBatches):
                if i == N_mBatches:
                    Batch = Data[:, idx[N_Samples_Train - mBS:]]
                else:
                    Batch = Data[:, idx[i * mBS:(i + 1) * mBS]]
                # Contrastive divergence updates
                RMSE = self.Mini_Batch_Update(Batch)
                RMSE_Train[N_mBatches * Epoch + i] = RMSE
                print("Epoch {:d}/{:d}, Batch {:{prec}d}/{:d}, RMSE={:.5f}".format(Epoch + 1, N_Epochs, i + 1, N_mBatches, RMSE, prec = N_Digs))
        self.PL = dict()
        self.PL["W"]   = self.W
        self.PL["b_v"] = self.b_v
        self.PL["b_h"] = self.b_h
        self.PL["sigma_v"] = self.sigma_v
        self.RMSE_Train = RMSE_Train
        return RMSE_Train

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def Mini_Batch_Update(self, v_0):
        d_W   = np.zeros([self.V, self.H])
        d_b_v = np.zeros([self.V, 1])
        d_b_h = np.zeros([self.H, 1])

        v_k = self.GibbsSampling(v_0)

        # Hyperbolic tangent terms which will be used in the CD-updates
        tanh_v_0 = np.tanh(np.matmul(self.W.transpose(), v_0 / self.sv_sq) + self.b_h)
        tanh_v_k = np.tanh(np.matmul(self.W.transpose(), v_k / self.sv_sq) + self.b_h)
        # Outer product using Einstein summation convention
        v_0_ts = np.expand_dims(v_0, axis=1)
        tanh_v_0_ts = np.expand_dims(tanh_v_0, axis=0)
        vtan_0 = np.einsum('aik,ijk->ajk', v_0_ts, tanh_v_0_ts)
        # Same for v_k
        v_k_ts = np.expand_dims(v_k, axis=1)
        tanh_v_k_ts = np.expand_dims(tanh_v_k, axis=0)
        vtan_k = np.einsum('aik,ijk->ajk', v_k_ts, tanh_v_k_ts)
        # Contrastive divergence updates
        d_W   = np.mean((vtan_0 - vtan_k) / self.sv_sq_ts, axis=2)
        d_b_v = np.mean((v_0 - v_k) / self.sv_sq, axis=1, keepdims=True)
        d_b_h = np.mean(tanh_v_0 - tanh_v_k, axis=1, keepdims=True)

        # Update all parameters
        self.D_W   = (1 - self.mu) * self.nu * d_W + self.mu * self.D_W
        self.W     = self.W + self.D_W

        self.D_b_v = (1 - self.mu) * self.nu * d_b_v + self.mu * self.D_b_v
        self.b_v   = self.b_v + self.D_b_v

        self.D_b_h = (1 - self.mu) * self.nu * d_b_h + self.mu * self.D_b_h
        self.b_h   = self.b_h + self.D_b_h
        RMSE = np.sqrt(np.mean(np.square((v_0 - v_k)), axis=(0,1)))
        return RMSE

    def Test(self, v):
        v_k = self.GibbsSampling(v)
        RMSE = np.sqrt(np.mean(np.square((v - v_k)), axis=(0,1)))
        print("=============== Test ===============")
        print("RMSE Test =", RMSE)
        return RMSE, v_k

    def GibbsSampling(self, v):
        # Probability P(h=-1|v)
        P_h_m1_v = self.sigmoid(-2 * (np.matmul(self.W.transpose(), v / self.sv_sq) + self.b_h))
        r = np.random.uniform(size=[self.H, 1], low = 0, high = +1)
        h = 2 * (r > P_h_m1_v) - 1
        # Sample Visibe Units
        v_k = self.b_v + np.matmul(self.W, h)
        return v_k

    def SaveModel(self, FileName = None):
        if FileName is None:
            FileName = "Model,V=%d,H=%d.mat" % (self.V, self.H)
        print("Saving GBPRBM model parameters to "+FileName)
        savemat(FileName, self.PL)