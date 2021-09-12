import os 
import pandas as pd
import numpy as np
import matplotlib.image as mpimg

def ImagesToCSV(test_filename = 'test', train_filename = 'train'):
    path = 'data/mnist_png'

    Xtest, Xtrain = [], []
    Ytest, Ytrain = [], []
    
    for subdir, dirs, files in os.walk(path):
        if files:
            newsudir = subdir.split('\\')
            label = newsudir[-1]
            frametype = newsudir[-2]
            print(label, frametype)
            for index, file in enumerate(files):
                path = subdir + '/' + file
                image = mpimg.imread(path)
                image = image.reshape(-1)
                if frametype == "training":
                    Xtrain.append(image)
                    Ytrain.append(label)
                elif frametype == 'testing':
                    Xtest.append(image)
                    Ytest.append(label)       
                    
    Xtest, Xtrain = np.array(Xtest), np.array(Xtrain)     
    Ytest, Ytrain = np.array(Ytest), np.array(Ytrain)

    dfXtrain = pd.DataFrame(Xtrain)
    dfYtrain = pd.DataFrame(Ytrain, columns=['label'])
    dftrainingset = dfXtrain.join(dfYtrain)

    dfXtest = pd.DataFrame(Xtest)
    dfYtest = pd.DataFrame(Ytest, columns=['label'])
    dftestset = dfXtest.join(dfYtest)

    path = r'data\\' + train_filename + '.csv'
    dftrainingset.to_csv (path, index = False, header=True)

    path = r'data\\' + test_filename + '.csv'
    dftestset.to_csv (path, index = False, header=True)

    print(Xtest.shape, Xtrain.shape)
    print(Ytest.shape, Ytrain.shape)

#ImagesToCSV('test_new', 'train_new')    



