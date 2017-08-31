def MNIST_Download():
    import os.path
    import urllib.request
    import gzip
    import numpy as np
    import scipy.io as sio

    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES  = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS  = 't10k-labels-idx1-ubyte.gz'

    # CHeck if file exists, if not - download it
    FileList = (TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS)
    for File in FileList:
        if not os.path.isfile(File):
            urllib.request.urlretrieve(SOURCE_URL + File, File)

    # Gunzip all files
    for File in FileList:
        raw, _ = os.path.splitext(File)
        if not os.path.isfile(raw):
            InFile = gzip.open(File, 'rb')
            OutFile = open(raw, 'wb')
            OutFile.write( InFile.read() )
            InFile.close()
            OutFile.close()

    # TRAIN IMAGES
    obj = open(TRAIN_IMAGES.strip(".gz"), 'rb')
    if np.fromfile(obj, dtype='>i4', count=1) != 2051:
        print("Magic number in {} is not correct!".format(TRAIN_IMAGES.strip(".gz")))
    N_Images = np.fromfile(obj, dtype='>i4', count=1)[0]
    N_Rows   = np.fromfile(obj, dtype='>i4', count=1)[0]
    N_Cols   = np.fromfile(obj, dtype='>i4', count=1)[0]
    trainData = np.zeros([N_Images, N_Rows*N_Cols])
    temp = np.zeros([N_Cols, N_Rows])
    for i in range(N_Images):
        for j in range(N_Rows):
            temp[:,j] = np.fromfile(obj, dtype='B', count=N_Cols)/255
        trainData[i, :] = np.squeeze(temp.reshape(N_Rows*N_Cols,1))
    obj.close()

    # TRAIN LABELS
    obj = open(TRAIN_LABELS.strip(".gz"), 'rb')
    if np.fromfile(obj, dtype='>i4', count=1) != 2049:
        print("Magic number in {} is not correct!".format(TRAIN_LABELS.strip(".gz")))
    N_Labels = np.fromfile(obj, dtype='>i4', count=1)[0]
    Labels   = np.fromfile(obj, dtype='B', count=N_Labels)
    obj.close()
    trainLabels = np.zeros([N_Labels, 10], )
    for k in range(N_Labels):
        trainLabels[k, Labels[k]] = 1
    sio.savemat("MNIST_Train_Medal_Normalized.mat", {'trainData': trainData, 'trainLabels': trainLabels}, do_compression=True)

    # TEST IMAGES
    obj = open(TEST_IMAGES.strip(".gz"), 'rb')
    if np.fromfile(obj, dtype='>i4', count=1) != 2051:
        print("Magic number in {} is not correct!".format(TEST_IMAGES.strip(".gz")))
    N_Images = np.fromfile(obj, dtype='>i4', count=1)[0]
    N_Rows   = np.fromfile(obj, dtype='>i4', count=1)[0]
    N_Cols   = np.fromfile(obj, dtype='>i4', count=1)[0]
    testData = np.zeros([N_Images, N_Rows*N_Cols])
    temp = np.zeros([N_Cols, N_Rows])
    for i in range(N_Images):
        for j in range(N_Rows):
            temp[:,j] = np.fromfile(obj, dtype='B', count=N_Cols)/255
        testData[i,:] = np.squeeze(temp.reshape(N_Rows*N_Cols,1))
    obj.close()

    # TEST LABELS
    obj = open(TEST_LABELS.strip(".gz"), 'rb')
    if np.fromfile(obj, dtype='>i4', count=1) != 2049:
        print("Magic number in {} is not correct!".format(TEST_LABELS.strip(".gz")))
    N_Labels = np.fromfile(obj, dtype='>i4', count=1)[0]
    Labels   = np.fromfile(obj, dtype='B', count=N_Labels)
    obj.close()
    testLabels = np.zeros([N_Labels, 10], )
    for k in range(N_Labels):
        testLabels[k, Labels[k]] = 1
    sio.savemat("MNIST_Test_Medal_Normalized.mat", {'testData': testData, 'testLabels': testLabels}, do_compression=True)

if __name__ == "__main__":
    MNIST_Download()