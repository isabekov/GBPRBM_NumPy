import numpy as np
import guiqwt.pyplot as plt

def MNIST_Plot_Image_Pairs(Img_Orig,Img_Recon, **kwargs):
    # This function plots pairs of MNIST images from two matrices in a grid
    # by reshaping 784x1 column vectors from supplied matrices into pairs of
    # two 28x28 images concatenated side by side.
    #                 Inputs
    # ============================================================
    # Img_Orig : 784 x N_Plots    First image
    # Img_Recon: 784 x N_Plots    Second image
    #                Optional inputs
    # N_Horiz  :   integer        Number of pairs of images on horizonal axis
    # N_Vert   :   integer        Number of pairs of images on vertical axis
    # str      :   string         Figure title
    #
    #                 Output
    # ============================================================
    # A figure with pairs of images

    if Img_Orig.shape != Img_Recon.shape:
        print('Error: Dimension mismatch!')

    if Img_Orig.shape[0] != 784:
        print('Error: Weight matrix does not have 784 rows!')

    if "str" in kwargs:
        str = kwargs["str"]
    else:
        str = 'MNIST: Pairs of images'

    # Number of images to reconstruct and plot
    N_Plots = Img_Orig.shape[1]

    if ("N_Horiz" in kwargs) and ("N_Vert" in kwargs):
        N_Horiz = kwargs["N_Horiz"]
        N_Vert  = kwargs["N_Vert"]
        if N_Horiz*N_Vert ==N_Plots:
            Do_Init = False
        else:
            Do_Init = True
    else:
        Do_Init=True

    if Do_Init==True:
        N_Horiz = int(np.floor(np.sqrt(N_Plots)))
        N_Vert = int(np.ceil(np.sqrt(N_Plots)))
        if N_Horiz*N_Vert < N_Plots:
           N_Horiz = N_Horiz + 1

    # Vertical padding
    V_Pad = 2 # pixels
    # Horizontal padding
    H_Pad = 2 # pixels
    # Number of pixels in width (height as well, this is a square image)
    N = 28 # pixels
    # Allocate memory for the combined image
    A = np.zeros([N*N_Horiz + H_Pad*(N_Horiz-1), 2*N*N_Vert + V_Pad*(N_Vert-1)])
    # Counter for the image sample to be plotted
    cnt = 0
    for i in range(N_Horiz):
        for j in range(N_Vert):
            # Calculating image position on the grid
            idx_H1 =     i*(N + H_Pad)
            idx_H2 = (i+1)*(N + H_Pad) - H_Pad
            idx_V1 =     j*(2*N + V_Pad)
            idx_V2 = (j+1)*(2*N + V_Pad) - V_Pad
            # Reshape a 784x1 vector into a 28x28 image. Do the same operation for the reconstructed image
            A[idx_H1:idx_H2, idx_V1:idx_V2] = np.concatenate([Img_Orig[:,cnt].reshape(N,N, order='F'), Img_Recon[:,cnt].reshape(N,N, order='F')], axis=1)
            cnt = cnt + 1
            if cnt == N_Plots:
                break
    plt.figure(str)
    plt.gray()  # colormap
    plt.imshow(A)
    plt.show()