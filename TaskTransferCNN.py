## --------- Best Improvement pipe: Tuned parameters _+ Data Augmentation-------------

import tensorflow
import cv2
import numpy as np
import os
import scipy.io as sio
import keras
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from keras.applications.resnet_v2 import resnet_v2
from keras.preprocessing import image
from keras.layers import Input, Flatten, Dense, Activation, merge
from keras.models import Model
from keras import optimizers
from keras.utils import np_utils
import time
from keras.applications.resnet_v2 import ResNet50V2, preprocess_input, decode_predictions
import pickle
import copy
from skimage.transform import rotate
from skimage.util import random_noise

np.random.seed(27)

def getParameters():
    """
    sets the initial parameters
    :return: dictionary of all the relevant parameters
    """
    s = 224
    t = 0.5
    lr = 0.01
    decay = 0.01
    momentum = 0
    params = {'S': s, 'T': t, 'LR': lr, 'Decay': decay, 'Momentum': momentum}
    return params


def InitData(path, params):
    """
    loads the data from the desired path
    :param path: the path of the data folders
    :param  params: dictionary of all the parameters
    :return: dictionary of 2 lists- Data (images) and the Labels (classes)
    """
    DandL = {'Data': [], 'Labels': []}
    dirsStrings = os.listdir(path)
    dirs = []
    for i in range(0, len(dirsStrings)-1):
        dirs.append(int(dirsStrings[i][:-5]))
    dirs = sorted(dirs)
    DandL['Labels'] = sio.loadmat(path + 'FlowerDataLabels.mat')
    DandL['Labels'] = DandL['Labels']['Labels']
    DandL['Labels'] = DandL['Labels'].reshape(DandL['Labels'].size, 1)
    s = params['S']
    for tempImageNum in dirs:
        img_path = path + '/' + str(tempImageNum) + '.jpeg'
        #img = cv2.imread(img_path)
        #img = cv2.resize(img, (s, s), interpolation=cv2.INTER_CUBIC)
        img = image.load_img(img_path, target_size=(s, s))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x.astype(float)
        x = preprocess_input(x)
        DandL['Data'].append(x)
    DandL['Data'] = np.array(DandL['Data'])
    DandL['Data'] = np.rollaxis(DandL['Data'], 1, 0)
    DandL['Data'] = DandL['Data'][0]
    print("Data loading completed")
    return DandL


def DataAugmentation(data, labels):
    """
    makes the data augmentation - for some of the images (a random choice) we make a rotation in random
    degree (from 10 to 90) and for some of the image we add a random noise.
    :param data: the data for making augmentation on
    :param  labels: the corresponding labels of the data
    :return: dictionary of 2 lists- Data (original images and their augmentation) and the Labels (classes)
    """
    augData = []
    augLabels =[]
    for i in range(0, len(data)):
        augManipulation = np.random.randint(1,3) # random int between 1 and 2
        if augManipulation == 1: # random noise
            randvariance = np.random.uniform(0.001, 0.01)
            img = random_noise(data[i], var=randvariance)
            augData.append(data[i])
            augData.append(img)
            augLabels.append(labels[i])
            augLabels.append(labels[i])
        if augManipulation == 2: # random flip
            randDegrees = np.random.randint(10,91)
            img = rotate(data[i], randDegrees)
            augData.append(data[i])
            augData.append(img)
            augLabels.append(labels[i])
            augLabels.append(labels[i])
    return augData, augLabels


def TrainTestSplit(data, labels):
    """
    splits the data into train data and test data with their respective labels, and send the training data
    and labels to the 'DataAugmentation'function.
    :param data: all the data (images) that has been loaded
    :param labels: all the labels of the data (classes)
    :return: dictionary of 4 lists- 2 lists for the train (data and labels) and 2 for the test
    """
    splitData = {'TrainData': [], 'TrainLabels': [], 'TestData': [], 'TestLabels': []}
    all_images_indices = range(1,473)
    train_images_indices = list(set(all_images_indices) - set(test_images_indices))
    zeroValueTrain_images_indices = [x - 1 for x in train_images_indices]
    zeroValueTest_images_indices = [x - 1 for x in test_images_indices]
    splitData['TrainData'] = data[zeroValueTrain_images_indices]
    splitData['TestData'] = data[zeroValueTest_images_indices]
    splitData['TrainLabels'] = labels[zeroValueTrain_images_indices]
    splitData['TestLabels'] = labels[zeroValueTest_images_indices]

    splitData['TrainData'], splitData['TrainLabels'] = DataAugmentation(splitData['TrainData'], splitData['TrainLabels'])

    splitData['TrainData'] = np.asarray(splitData['TrainData'])
    splitData['TestData'] = np.asarray(splitData['TestData'])
    splitData['TrainLabels'] = np.asarray(splitData['TrainLabels'])
    splitData['TestLabels'] = np.asarray(splitData['TestLabels'])

    return splitData


def Train(SplitData, params):
    """
    Train the last layer of the network over the train data
    :param SplitData: a dict of the 4 lists - 2 for the train (data+labels) and 2 for
    the test (data+labels)
    :return: the trained model
    """
    print("Start Training")
    trainData, trainLabels = SplitData['TrainData'], SplitData['TrainLabels']

    model = keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet', pooling= 'avg')

    our_last_layer = model.get_layer('avg_pool').output
    out = Dense(1, activation='sigmoid', name='output')(our_last_layer)
    custom_resNet_model = Model(model.input, out)

    # set all the layers except the last to be not trainable
    for layer in custom_resNet_model.layers[:-1]:
        layer.trainable = False

    #custom_resNet_model.summary()
    opt = optimizers.SGD(lr=params['LR'], decay=params['Decay'], momentum=params['Momentum'])
    custom_resNet_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    t = time.time()
    custom_resNet_model.fit(trainData, trainLabels, batch_size=32, epochs=10, verbose=0)
    trainingTime = (time.time()-t)//60
    print('Training completed after : ' + str(trainingTime) + ' minutes')
    return custom_resNet_model


def Test(model, testData):
    """
    makes the model predictions of the test data
    :param model: the trained model
    :param testData: the test data
    :return: the predictions (probability to be a flower) for each test image
    """
    results = model.predict(testData)
    print("Test completed")
    return results


def Evaluate(results, splitData, params):
    """
    evaluates the model results by the truth labels and finds the largest errors
    :param results: the predictions of the model (probability to be a flower)
    :param splitData: 4 list - a dict of the 4 lists - 2 for the train (data+labels) and 2 for
    the test (data+labels)
    :param params: a dict of all the parameters
    :return: a dict of the error rate, type 1 error list and type 2 error list
    """
    testData, testLabels = splitData['TestData'], SplitData['TestLabels']
    predictions = results > params['T']
    sum = 0
    for i in range(0, len(predictions)):
        if predictions[i] == testLabels[i]:
            sum = sum + 1
    errorRate = 1-(sum/len(predictions))
    type1Errors = {'Indices': [], 'Results': []}
    type2Errors = {'Indices': [], 'Results': []}
    for i in range(0, len(predictions)):
        imageIndex = test_images_indices[i]
        if (predictions[i] == 0) and (testLabels[i] == 1):
            type1Errors['Indices'].append(imageIndex)
            type1Errors['Results'].append(results[i])
        if (predictions[i] == 1) and (testLabels[i] == 0):
            type2Errors['Indices'].append(imageIndex)
            type2Errors['Results'].append(results[i])
    topType1Errors = {'Indices': [], 'Results': []}
    topType2Errors = {'Indices': [], 'Results': []}
    for k in range(0,5):
        if len(type1Errors['Indices']) > 0:
            localMinIndex = type1Errors['Results'].index(min(type1Errors['Results']))
            globalMinIndex = type1Errors['Indices'][localMinIndex]
            MinError = type1Errors['Results'][localMinIndex]
            topType1Errors['Indices'].append(globalMinIndex)
            topType1Errors['Results'].append(MinError)

            type1Errors['Results'].pop(localMinIndex)
            type1Errors['Indices'].pop(localMinIndex)
        if len(type2Errors['Indices']) > 0:
            localMinIndex = type2Errors['Results'].index(max(type2Errors['Results']))
            globalMinIndex = type2Errors['Indices'][localMinIndex]
            MinError = type2Errors['Results'][localMinIndex]
            topType2Errors['Indices'].append(globalMinIndex)
            topType2Errors['Results'].append(MinError)

            type2Errors['Results'].pop(localMinIndex)
            type2Errors['Indices'].pop(localMinIndex)
    print("Evaluate completed")
    return {'ErrorRate': errorRate, 'Type1': topType1Errors, 'Type2': topType2Errors}


def ReportResults(Summary, Results):
    """
    prints the required report including the test error, the error lists and
    the precision recall curve
    :param Summary: a dict of the error rate, type 1 error list and type 2 error list
    :param Results: the predictions of the model (probability to be a flower)
    """
    print("Test Error: " + str(Summary['ErrorRate']))
    for i in range(0, len(Summary['Type1']['Indices'])):
        index = Summary['Type1']['Indices'][i]
        print("Error Type 1, Error Index:  " + str(i + 1) + ", Image Name: " + str(index) + ", Classification Score: " + str(
            Summary['Type1']['Results'][i]))
        img = cv2.imread(data_path + '/' + str(index) + '.jpeg')
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #plt.imshow(rgbImg)
        #plt.title("Error Type 1, Error Index:  " + str(i+1) + ", Classification Score: "
                  #+ str(Summary['Type1']['Results'][i]))
        #plt.show()
    for i in range(0, len(Summary['Type2']['Indices'])):
        index = Summary['Type2']['Indices'][i]
        print("Error Type 2, Error Index:  " + str(i + 1) + ", Image Name: " + str(index) + ", Classification Score: " + str(
            Summary['Type2']['Results'][i]))
        img = cv2.imread(data_path + '/' + str(index) + '.jpeg')
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #plt.imshow(rgbImg)
        #plt.title("Error Type 2, Error Index:  " + str(i+1) + ", Classification Score: "
                 #+ str(Summary['Type2']['Results'][i]))
        #plt.show()

    testLabels= []
    for i in range(0, len(SplitData['TestLabels'])):
        testLabels.append(SplitData['TestLabels'][i][0])
    scores = []
    for i in range(0, len(Results)):
        scores.append(Results[i][0])
    precision, recall, thresholds = precision_recall_curve(testLabels, scores)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()


def EqualsDicts(dict1, dict2):
    """
    compares two dictionaries and decides whether they equals
    :param dict1: a dictionary of all the parameters
    :param dict2: a dictionary of all the parameters
    :return: boolean whether both of the dictionaries are equals
    """
    flag = True
    if dict1['T'] != dict2['T']:
        flag = False
    if dict1['LR'] != dict2['LR']:
        flag = False
    if dict1['Decay'] != dict2['Decay']:
        flag = False
    if dict1['Momentum'] != dict2['Momentum']:
        flag = False
    return flag


def ValidationEvaluate(results, tuningData, params):
    """
    evaluates the model for validation results by the truth labels and finds the largest errors
    :param results: the predictions of the model (probability to be a flower)
    :param tuningData: 4 list - a dict of the 4 lists - 2 for the train (data+labels) and 2 for
    the validation (data+labels)
    :param params: a dict of all the parameters
    :return: the validation error rate
     """
    valData, valLabels = tuningData['ValData'], tuningData['ValLabels']
    predictions = results > params['T']
    sum = 0
    for i in range(0, len(predictions)):
        if predictions[i] == valLabels[i]:
            sum = sum + 1
    errorRate = 1 - (sum / len(predictions))
    return errorRate


def findValError(tuningData, params):
    """
    finds the validation error given the current params
    :param tuningData: 4 list - a dict of the 4 lists - 2 for the train (data+labels) and 2 for
    the validation (data+labels)
    :param params: a dict of all the parameters
    :return: the validation error rate (by calling 'ValidationEvaluate' function)
     """
    model = Train(tuningData, params)
    results = Test(model, tuningData['ValData'])
    valError = ValidationEvaluate(results, tuningData, params)
    return valError


def PrintPlots(tValuesAndErrors, lrValuesAndErrors, decayValuesAndErrors, momentumValuesAndErrors):
    """
    prints the  plots for the parameters tuning
    :param tValuesAndErrors: the validation error per each of the checked values for parameter t
    :param lrValuesAndErrors: the validation error per each of the checked values for parameter learning rate
    :param decayValuesAndErrors: the validation error per each of the checked values for parameter decay
    :param momentumValuesAndErrors: the validation error per each of the checked values for parameter momentum
     """
    plt.plot(tValuesAndErrors['Values'], tValuesAndErrors['Errors'])
    plt.xlabel('T values')
    plt.ylabel('Validation Error')
    plt.savefig('T.png')
    plt.close()

    plt.semilogx(lrValuesAndErrors['Values'], lrValuesAndErrors['Errors'])
    plt.xlabel('LR values')
    plt.ylabel('Validation Error')
    plt.savefig('LR_log.png')
    plt.close()

    plt.plot(lrValuesAndErrors['Values'], lrValuesAndErrors['Errors'])
    plt.xlabel('LR values')
    plt.ylabel('Validation Error')
    plt.savefig('LR.png')
    plt.close()

    plt.semilogx(decayValuesAndErrors['Values'], decayValuesAndErrors['Errors'])
    plt.xlabel('Decay values')
    plt.ylabel('Validation Error')
    plt.savefig('Decay_log.png')
    plt.close()

    plt.plot(decayValuesAndErrors['Values'], decayValuesAndErrors['Errors'])
    plt.xlabel('Decay values')
    plt.ylabel('Validation Error')
    plt.savefig('Decay.png')
    plt.close()

    plt.plot(momentumValuesAndErrors['Values'], momentumValuesAndErrors['Errors'])
    plt.xlabel('Momentum values')
    plt.ylabel('Validation Error')
    plt.savefig(str(type) + ' Momentum.png')
    plt.close()


def Tuning(trainData, trainLabels):
    """
    makes the tuning for relevant parameters and finds their best values which minimized the validation error
    :param trainData: list of the train data (images)
    :param trainLabels: list of the respective train labels
    :return: a dictionary with the best parameters that were found
     """
    TuningData = {'TrainData': [], 'TrainLabels': [], 'ValData': [], 'ValLabels': []}
    splitIndex = round(0.8*len(trainData))
    TuningData['TrainData'] = trainData[0:splitIndex]
    TuningData['TrainLabels'] = trainLabels[0:splitIndex]
    TuningData['ValData'] = trainData[splitIndex:]
    TuningData['ValLabels'] = trainLabels[splitIndex:]
    s = 224
    params = {'S': s, 'T': 0.5, 'LR': 0.01, 'Decay': 0.01, 'Momentum': 0.5}
    currentParams = copy.deepcopy(params)
    print("Initial:")
    print("Current Params = " + str(currentParams))
    for iterarion in range(0, 5):
        previousParams = copy.deepcopy(currentParams)
        print("Start Iteration:")
        print("Previous Params = " + str(previousParams))
        print("Current Params = " + str(currentParams))
        tValuesAndErrors = {'Values': np.arange(0.5, 0.91, 0.1), 'Errors': []}
        for t in tValuesAndErrors['Values']:
            currentParams['T'] = t
            ValError = findValError(TuningData, currentParams)
            tValuesAndErrors['Errors'].append(ValError)
        minErrorIndex = tValuesAndErrors['Errors'].index(min(tValuesAndErrors['Errors']))
        sMinError = tValuesAndErrors['Values'][minErrorIndex]
        currentParams['T'] = sMinError
        print("best t for type " + str(type) + " is " + str(currentParams['T']))

        lrValuesAndErrors = {'Values': np.logspace(-4, 0, 5), 'Errors': []}
        for lr in lrValuesAndErrors['Values']:
            currentParams['LR'] = lr
            ValError = findValError(TuningData, currentParams)
            lrValuesAndErrors['Errors'].append(ValError)
        minErrorIndex = lrValuesAndErrors['Errors'].index(min(lrValuesAndErrors['Errors']))
        lrMinError = lrValuesAndErrors['Values'][minErrorIndex]
        currentParams['LR'] = lrMinError
        print("best lr for type " + str(type) + " is " + str(currentParams['LR']))

        decayValuesAndErrors = {'Values': [0, 0.001, 0.01, 0.1, 1], 'Errors': []}
        for decay in decayValuesAndErrors['Values']:
            currentParams['Decay'] = decay
            ValError = findValError(TuningData, currentParams)
            decayValuesAndErrors['Errors'].append(ValError)
        minErrorIndex = decayValuesAndErrors['Errors'].index(min(decayValuesAndErrors['Errors']))
        decayMinError = decayValuesAndErrors['Values'][minErrorIndex]
        currentParams['Decay'] = decayMinError
        print("best decay for type " + str(type) + " is " + str(currentParams['Decay']))

        momentumValuesAndErrors = {'Values': [0, 0.1, 0.3, 0.5, 0.7, 0.9], 'Errors': []}
        for momentum in momentumValuesAndErrors['Values']:
            currentParams['Momentum'] = momentum
            ValError = findValError(TuningData, currentParams)
            momentumValuesAndErrors['Errors'].append(ValError)
        minErrorIndex = momentumValuesAndErrors['Errors'].index(min(momentumValuesAndErrors['Errors']))
        momentumMinError = momentumValuesAndErrors['Values'][minErrorIndex]
        currentParams['Momentum'] = momentumMinError
        print("best momentum for type " + str(type) + " is " + str(currentParams['Momentum']))

        if EqualsDicts(currentParams, previousParams):
            break
    PrintPlots(tValuesAndErrors, lrValuesAndErrors, decayValuesAndErrors, momentumValuesAndErrors)
    return currentParams

def setBestParamsAfterTuning(params):
    """
    this function created for it will be not necessary to run again the tuning function (which takes
    a lot of time), and it sets the best values the tuning function found.
    :param params: dictionary of all the parameters
    :return: a dictionary with the best parameters that were found
     """
    params['T'] = 0.8
    params['LR'] = 0.01
    params['Decay'] = 0.1
    params['Momentum'] = 0.5
    return params


#----------Main------------------------------------
startTime = time.time()
data_path = "C:/Users/wr/Downloads/FlowerData/"
test_images_indices = list(range(301, 473))
Params = getParameters()
DandL = InitData(data_path, Params)
SplitData = TrainTestSplit(DandL['Data'], DandL['Labels'])
#Params = Tuning(SplitData['TrainData'], SplitData['TrainLabels'])
Params = setBestParamsAfterTuning(Params)
Model = Train(SplitData, Params)
#pickle.dump(Model, open( "save_improve5_model.p", "wb" ))
#Model = pickle.load( open( "save_improve5_model.p", "rb" ) )
Results = Test(Model, SplitData['TestData'])
#pickle.dump(Results, open( "save_improve5_results.p", "wb" ))
#Results = pickle.load( open( "save_improve5_results.p", "rb" ) )
Summary = Evaluate(Results, SplitData, Params)
ReportResults(Summary, Results)
totalTime = (time.time()-startTime)//60
print("Total Time: " + str(totalTime) + " minutes")


''''' 
# ------ Basic CNN----------------------------------------------------------------

def getParameters():
    s = 224
    params = {'S': s}
    return params


def InitData(path, params):
    """
    loads the data from the desired path
    :param path: the path of the data folders
    :param  params: dictionary of all the parameters
    :return: dictionary of 2 lists- Data (images) and the Labels (classes)
    """
    DandL = {'Data': [], 'Labels': []}
    dirsStrings = os.listdir(path)
    dirs = []
    for i in range(0, len(dirsStrings)-1):
        dirs.append(int(dirsStrings[i][:-5]))
    dirs = sorted(dirs)
    DandL['Labels'] = sio.loadmat(path + 'FlowerDataLabels.mat')
    DandL['Labels'] = DandL['Labels']['Labels']
    DandL['Labels'] = DandL['Labels'].reshape(DandL['Labels'].size, 1)
    s = params['S']
    for tempImageNum in dirs:
        img_path = path + '/' + str(tempImageNum) + '.jpeg'
        #img = cv2.imread(img_path)
        #img = cv2.resize(img, (s, s), interpolation=cv2.INTER_CUBIC)
        img = image.load_img(img_path, target_size=(s, s))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x.astype(float)
        x = preprocess_input(x)
        DandL['Data'].append(x)
    DandL['Data'] = np.array(DandL['Data'])
    DandL['Data'] = np.rollaxis(DandL['Data'], 1, 0)
    DandL['Data'] = DandL['Data'][0]
    return DandL


def TrainTestSplit(data, labels):
    """
    splits the data into train data and test data with their respective labels
    :param data: all the data (images) that has been loaded
    :param labels: all the labels of the data (classes)
    :return: dictionary of 4 lists- 2 lists for the train (data and labels) and 2 for the test
    """
    splitData = {'TrainData': [], 'TrainLabels': [], 'TestData': [], 'TestLabels': []}
    all_images_indices = range(1,473)
    train_images_indices = list(set(all_images_indices) - set(test_images_indices))
    zeroValueTrain_images_indices = [x - 1 for x in train_images_indices]
    zeroValueTest_images_indices = [x - 1 for x in test_images_indices]
    splitData['TrainData'] = data[zeroValueTrain_images_indices]
    splitData['TestData'] = data[zeroValueTest_images_indices]
    splitData['TrainLabels'] = labels[zeroValueTrain_images_indices]
    splitData['TestLabels'] = labels[zeroValueTest_images_indices]
    splitData['TrainData'] = np.asarray(splitData['TrainData'])
    splitData['TestData'] = np.asarray(splitData['TestData'])
    splitData['TrainLabels'] = np.asarray(splitData['TrainLabels'])
    splitData['TestLabels'] = np.asarray(splitData['TestLabels'])

    return splitData


def Train(SplitData):
    """
    Train the last layer of the network over the train data
    :param SplitData: a dict of the data (images+labels) splits to train and test, will get the value returned from TrainTestSplit function
    :return: the trained model
    """
    trainData, trainLabels = SplitData['TrainData'], SplitData['TrainLabels']

    model = keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet', pooling= 'avg')

    our_last_layer = model.get_layer('avg_pool').output
    out = Dense(1, activation='sigmoid', name='output')(our_last_layer)
    custom_resNet_model = Model(model.input, out)


    # freeze the layers so we don't train them
    for layer in custom_resNet_model.layers[:-1]:
        layer.trainable = False

    #custom_resNet_model.summary()
    custom_resNet_model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    t = time.time()
    custom_resNet_model.fit(trainData, trainLabels, batch_size=32, epochs=10, verbose=0)
    print('Training time: %s' % (time.time()-t))

    return custom_resNet_model


def Test(model, testData):
    """
    Implementing our trained classifying model over the test data
    :param model: the trained model
    :param testData: array of the test data
    :return: the predictions (the CNN score - probability to be a flower) for each test sample
    """
    results = model.predict(testData)
    return results


def Evaluate(results, splitData):
    testData, testLabels = splitData['TestData'], SplitData['TestLabels']
    predictions = results > 0.5
    summ = np.sum(splitData['TrainLabels'])
    sum = 0
    for i in range(0, len(predictions)):
        if predictions[i] == testLabels[i]:
            sum = sum + 1
    errorRate = 1-(sum/len(predictions))
    type1Errors = {'Indices': [], 'Results': []}
    type2Errors = {'Indices': [], 'Results': []}
    for i in range(0, len(predictions)):
        imageIndex = test_images_indices[i]
        if (predictions[i] == 0) and (testLabels[i] == 1):
            type1Errors['Indices'].append(imageIndex)
            type1Errors['Results'].append(results[i])
        if (predictions[i] == 1) and (testLabels[i] == 0):
            type2Errors['Indices'].append(imageIndex)
            type2Errors['Results'].append(results[i])
    print(type1Errors['Indices'])
    print(type2Errors['Indices'])
    topType1Errors = {'Indices': [], 'Results': []}
    topType2Errors = {'Indices': [], 'Results': []}
    for k in range(0,5):
        if len(type1Errors['Indices']) > 0:
            localMinIndex = type1Errors['Results'].index(min(type1Errors['Results']))
            globalMinIndex = type1Errors['Indices'][localMinIndex]
            MinError = type1Errors['Results'][localMinIndex]
            topType1Errors['Indices'].append(globalMinIndex)
            topType1Errors['Results'].append(MinError)

            type1Errors['Results'].pop(localMinIndex)
            type1Errors['Indices'].pop(localMinIndex)
        if len(type2Errors['Indices']) > 0:
            localMinIndex = type2Errors['Results'].index(max(type2Errors['Results']))
            globalMinIndex = type2Errors['Indices'][localMinIndex]
            MinError = type2Errors['Results'][localMinIndex]
            topType2Errors['Indices'].append(globalMinIndex)
            topType2Errors['Results'].append(MinError)

            type2Errors['Results'].pop(localMinIndex)
            type2Errors['Indices'].pop(localMinIndex)
    print(topType1Errors['Indices'])
    print(topType2Errors['Indices'])
    return {'ErrorRate': errorRate, 'Type1': topType1Errors, 'Type2': topType2Errors}





def ReportResults(Summary, Results):
    print("Test Error: " + str(Summary['ErrorRate']))
    for i in range(0, len(Summary['Type1']['Indices'])):
        index = Summary['Type1']['Indices'][i]
        img = cv2.imread(data_path + '/' + str(index) + '.jpeg')
        plt.imshow(img)
        plt.title("Error Type 1, Error Index:  " + str(i+1) + ", Classification Score: "
                  + str(Summary['Type1']['Results'][i]))
        plt.show()
    for i in range(0, len(Summary['Type2']['Indices'])):
        index = Summary['Type2']['Indices'][i]
        img = cv2.imread(data_path + '/' + str(i+1) + '.jpeg')
        plt.imshow(img)
        plt.title("Error Type 2, Error Index:  " + str(index) + ", Classification Score: "
                  + str(Summary['Type2']['Results'][i]))
        plt.show()

    testLabels= []
    for i in range(0, len(SplitData['TestLabels'])):
        testLabels.append(SplitData['TestLabels'][i][0])
    scores = []
    for i in range(0, len(Results)):
        scores.append(Results[i][0])
    print(scores)
    precision, recall, thresholds = precision_recall_curve(testLabels, scores)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()


#----------Main_Basic_CNN------------------

data_path = "C:/Users/wr/Downloads/FlowerData/"
test_images_indices = list(range(301, 473))
Params = getParameters()
DandL = InitData(data_path, Params)
SplitData = TrainTestSplit(DandL['Data'], DandL['Labels'])
Model = Train(SplitData)
#pickle.dump(Model, open( "save_model.p", "wb" ))
#Model = pickle.load( open( "save_model.p", "rb" ) )
#history = fitModel(Model, SplitData['TrainData'], SplitData['TrainLabels'])
Results = Test(Model, SplitData['TestData'])
#pickle.dump(Results, open( "save_results.p", "wb" ))
#Results = pickle.load( open( "save_results.p", "rb" ) )
Summary = Evaluate(Results, SplitData)
ReportResults(Summary, Results)

# ------ Improvement 1: Data Augmentation-----------------------------------------

def getParameters():
    s = 224
    params = {'S': s}
    return params


def InitData(path, params):
    """
    loads the data from the desired path
    :param path: the path of the data folders
    :param  params: dictionary of all the parameters
    :return: dictionary of 2 lists- Data (images) and the Labels (classes)
    """
    DandL = {'Data': [], 'Labels': []}
    dirsStrings = os.listdir(path)
    dirs = []
    for i in range(0, len(dirsStrings)-1):
        dirs.append(int(dirsStrings[i][:-5]))
    dirs = sorted(dirs)
    DandL['Labels'] = sio.loadmat(path + 'FlowerDataLabels.mat')
    DandL['Labels'] = DandL['Labels']['Labels']
    DandL['Labels'] = DandL['Labels'].reshape(DandL['Labels'].size, 1)
    s = params['S']
    for tempImageNum in dirs:
        img_path = path + '/' + str(tempImageNum) + '.jpeg'
        #img = cv2.imread(img_path)
        #img = cv2.resize(img, (s, s), interpolation=cv2.INTER_CUBIC)
        img = image.load_img(img_path, target_size=(s, s))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x.astype(float)
        x = preprocess_input(x)
        DandL['Data'].append(x)
    DandL['Data'] = np.array(DandL['Data'])
    DandL['Data'] = np.rollaxis(DandL['Data'], 1, 0)
    DandL['Data'] = DandL['Data'][0]
    return DandL


def DataAugmentation(data, labels):
    augData = []
    augLabels =[]
    for i in range(0, len(data)):
        augManipulation = np.random.randint(1,3) # random int between 1 and 2
        if augManipulation == 1: # random noise
            randvariance = np.random.uniform(0.001, 0.01)
            img = random_noise(data[i], var=randvariance)
            augData.append(data[i])
            augData.append(img)
            augLabels.append(labels[i])
            augLabels.append(labels[i])
        if augManipulation == 2: # random flip
            randDegrees = np.random.randint(10,91)
            img = rotate(data[i], randDegrees)
            augData.append(data[i])
            augData.append(img)
            augLabels.append(labels[i])
            augLabels.append(labels[i])
    return augData, augLabels


def TrainTestSplit(data, labels):
    """
    splits the data into train data and test data with their respective labels
    :param data: all the data (images) that has been loaded
    :param labels: all the labels of the data (classes)
    :return: dictionary of 4 lists- 2 lists for the train (data and labels) and 2 for the test
    """
    splitData = {'TrainData': [], 'TrainLabels': [], 'TestData': [], 'TestLabels': []}
    all_images_indices = range(1,473)
    train_images_indices = list(set(all_images_indices) - set(test_images_indices))
    zeroValueTrain_images_indices = [x - 1 for x in train_images_indices]
    zeroValueTest_images_indices = [x - 1 for x in test_images_indices]
    splitData['TrainData'] = data[zeroValueTrain_images_indices]
    splitData['TestData'] = data[zeroValueTest_images_indices]
    splitData['TrainLabels'] = labels[zeroValueTrain_images_indices]
    splitData['TestLabels'] = labels[zeroValueTest_images_indices]

    splitData['TrainData'], splitData['TrainLabels'] = DataAugmentation(splitData['TrainData'], splitData['TrainLabels'])

    splitData['TrainData'] = np.asarray(splitData['TrainData'])
    splitData['TestData'] = np.asarray(splitData['TestData'])
    splitData['TrainLabels'] = np.asarray(splitData['TrainLabels'])
    splitData['TestLabels'] = np.asarray(splitData['TestLabels'])


    return splitData


def Train(SplitData):
    """
    Train the last layer of the network over the train data
    :param SplitData: a dict of the data (images+labels) splits to train and test, will get the value returned from TrainTestSplit function
    :return: the trained model
    """
    trainData, trainLabels = SplitData['TrainData'], SplitData['TrainLabels']

    tensor_input = Input(shape=(224, 224, 3))
    model = keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet', pooling= 'avg')
    #model.summary()

    our_last_layer = model.get_layer('avg_pool').output
    out = Dense(1, activation='sigmoid', name='output')(our_last_layer)
    custom_resNet_model = Model(model.input, out)


    # freeze the layers so we don't train them
    for layer in custom_resNet_model.layers[:-1]:
        layer.trainable = False

    #custom_resNet_model.summary()
    custom_resNet_model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    t = time.time()
    custom_resNet_model.fit(trainData, trainLabels, batch_size=32, epochs=10, verbose=0)
    print('Training time: %s' % (time.time()-t))

    return custom_resNet_model



def Test(model, testData):
    """
    Implementing our trained classifying model over the test data
    :param model: the trained model
    :param testData: array of the test data
    :return: the predictions (the CNN score - probability to be a flower) for each test sample
    """
    results = model.predict(testData)
    return results


def Evaluate(results, splitData):
    testData, testLabels = splitData['TestData'], SplitData['TestLabels']
    predictions = results > 0.5
    summ = np.sum(splitData['TrainLabels'])
    sum = 0
    for i in range(0, len(predictions)):
        if predictions[i] == testLabels[i]:
            sum = sum + 1
    errorRate = 1-(sum/len(predictions))
    type1Errors = {'Indices': [], 'Results': []}
    type2Errors = {'Indices': [], 'Results': []}
    for i in range(0, len(predictions)):
        imageIndex = test_images_indices[i]
        if (predictions[i] == 0) and (testLabels[i] == 1):
            type1Errors['Indices'].append(imageIndex)
            type1Errors['Results'].append(results[i])
        if (predictions[i] == 1) and (testLabels[i] == 0):
            type2Errors['Indices'].append(imageIndex)
            type2Errors['Results'].append(results[i])
    print(type1Errors['Indices'])
    print(type2Errors['Indices'])
    topType1Errors = {'Indices': [], 'Results': []}
    topType2Errors = {'Indices': [], 'Results': []}
    for k in range(0,5):
        if len(type1Errors['Indices']) > 0:
            localMinIndex = type1Errors['Results'].index(min(type1Errors['Results']))
            globalMinIndex = type1Errors['Indices'][localMinIndex]
            MinError = type1Errors['Results'][localMinIndex]
            topType1Errors['Indices'].append(globalMinIndex)
            topType1Errors['Results'].append(MinError)

            type1Errors['Results'].pop(localMinIndex)
            type1Errors['Indices'].pop(localMinIndex)
        if len(type2Errors['Indices']) > 0:
            localMinIndex = type2Errors['Results'].index(max(type2Errors['Results']))
            globalMinIndex = type2Errors['Indices'][localMinIndex]
            MinError = type2Errors['Results'][localMinIndex]
            topType2Errors['Indices'].append(globalMinIndex)
            topType2Errors['Results'].append(MinError)

            type2Errors['Results'].pop(localMinIndex)
            type2Errors['Indices'].pop(localMinIndex)
    print(topType1Errors['Indices'])
    print(topType2Errors['Indices'])
    return {'ErrorRate': errorRate, 'Type1': topType1Errors, 'Type2': topType2Errors}





def ReportResults(Summary, Results):
    print("Test Error: " + str(Summary['ErrorRate']))
    for i in range(0, len(Summary['Type1']['Indices'])):
        index = Summary['Type1']['Indices'][i]
        img = cv2.imread(data_path + '/' + str(index) + '.jpeg')
        plt.imshow(img)
        plt.title("Error Type 1, #" + str(i+1) + " ,Image Name: " + str(index) + ", Classification Score: "
                  + str(Summary['Type1']['Results'][i]))
        plt.show()
    for i in range(0, len(Summary['Type2']['Indices'])):
        index = Summary['Type2']['Indices'][i]
        img = cv2.imread(data_path + '/' + str(i+1) + '.jpeg')
        plt.imshow(img)
        plt.title("Error Type 2, #" + str(i+1) + " ,Image Name: " + str(index) + ", Classification Score: "
                  + str(Summary['Type2']['Results'][i]))
        plt.show()

    testLabels= []
    for i in range(0, len(SplitData['TestLabels'])):
        testLabels.append(SplitData['TestLabels'][i][0])
    scores = []
    for i in range(0, len(Results)):
        scores.append(Results[i][0])
    print(scores)
    precision, recall, thresholds = precision_recall_curve(testLabels, scores)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()


#----------Main_Improve1------------------

data_path = "C:/Users/wr/Downloads/FlowerData/"
test_images_indices = list(range(301, 473))
Params = getParameters()
DandL = InitData(data_path, Params)
SplitData = TrainTestSplit(DandL['Data'], DandL['Labels'])
Model = Train(SplitData)
#pickle.dump(Model, open( "save_improve1_model.p", "wb" ))
#Model = pickle.load( open( "save_improve1_model.p", "rb" ) )
#history = fitModel(Model, SplitData['TrainData'], SplitData['TrainLabels'])
Results = Test(Model, SplitData['TestData'])
#pickle.dump(Results, open( "save_improve1_results.p", "wb" ))
#Results = pickle.load( open( "save_improve1_results.p", "rb" ) )
Summary = Evaluate(Results, SplitData)
ReportResults(Summary, Results)


# ------ Improvement 2: Data Augmentation + Max Pooling---------------------------

def getParameters():
    s = 224
    params = {'S': s}
    return params


def InitData(path, params):
    """
    loads the data from the desired path
    :param path: the path of the data folders
    :param  params: dictionary of all the parameters
    :return: dictionary of 2 lists- Data (images) and the Labels (classes)
    """
    DandL = {'Data': [], 'Labels': []}
    dirsStrings = os.listdir(path)
    dirs = []
    for i in range(0, len(dirsStrings)-1):
        dirs.append(int(dirsStrings[i][:-5]))
    dirs = sorted(dirs)
    DandL['Labels'] = sio.loadmat(path + 'FlowerDataLabels.mat')
    DandL['Labels'] = DandL['Labels']['Labels']
    DandL['Labels'] = DandL['Labels'].reshape(DandL['Labels'].size, 1)
    s = params['S']
    for tempImageNum in dirs:
        img_path = path + '/' + str(tempImageNum) + '.jpeg'
        #img = cv2.imread(img_path)
        #img = cv2.resize(img, (s, s), interpolation=cv2.INTER_CUBIC)
        img = image.load_img(img_path, target_size=(s, s))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x.astype(float)
        x = preprocess_input(x)
        DandL['Data'].append(x)
    DandL['Data'] = np.array(DandL['Data'])
    DandL['Data'] = np.rollaxis(DandL['Data'], 1, 0)
    DandL['Data'] = DandL['Data'][0]
    return DandL


def DataAugmentation(data, labels):
    augData = []
    augLabels =[]
    for i in range(0, len(data)):
        augManipulation = np.random.randint(1,3) # random int between 1 and 2
        if augManipulation == 1: # random noise
            randvariance = np.random.uniform(0.001, 0.01)
            img = random_noise(data[i], var=randvariance)
            augData.append(data[i])
            augData.append(img)
            augLabels.append(labels[i])
            augLabels.append(labels[i])
        if augManipulation == 2: # random flip
            randDegrees = np.random.randint(10,91)
            img = rotate(data[i], randDegrees)
            augData.append(data[i])
            augData.append(img)
            augLabels.append(labels[i])
            augLabels.append(labels[i])
    return augData, augLabels


def TrainTestSplit(data, labels):
    """
    splits the data into train data and test data with their respective labels
    :param data: all the data (images) that has been loaded
    :param labels: all the labels of the data (classes)
    :return: dictionary of 4 lists- 2 lists for the train (data and labels) and 2 for the test
    """
    splitData = {'TrainData': [], 'TrainLabels': [], 'TestData': [], 'TestLabels': []}
    all_images_indices = range(1,473)
    train_images_indices = list(set(all_images_indices) - set(test_images_indices))
    zeroValueTrain_images_indices = [x - 1 for x in train_images_indices]
    zeroValueTest_images_indices = [x - 1 for x in test_images_indices]
    splitData['TrainData'] = data[zeroValueTrain_images_indices]
    splitData['TestData'] = data[zeroValueTest_images_indices]
    splitData['TrainLabels'] = labels[zeroValueTrain_images_indices]
    splitData['TestLabels'] = labels[zeroValueTest_images_indices]

    splitData['TrainData'], splitData['TrainLabels'] = DataAugmentation(splitData['TrainData'], splitData['TrainLabels'])

    splitData['TrainData'] = np.asarray(splitData['TrainData'])
    splitData['TestData'] = np.asarray(splitData['TestData'])
    splitData['TrainLabels'] = np.asarray(splitData['TrainLabels'])
    splitData['TestLabels'] = np.asarray(splitData['TestLabels'])

    return splitData


def Train(SplitData):
    """
    Train the last layer of the network over the train data
    :param SplitData: a dict of the data (images+labels) splits to train and test, will get the value returned from TrainTestSplit function
    :return: the trained model
    """
    trainData, trainLabels = SplitData['TrainData'], SplitData['TrainLabels']

    tensor_input = Input(shape=(224, 224, 3))
    model = keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet', pooling= 'max')
    #model.summary()

    our_last_layer = model.get_layer('max_pool').output
    out = Dense(1, activation='sigmoid', name='output')(our_last_layer)
    custom_resNet_model = Model(model.input, out)


    # freeze the layers so we don't train them
    for layer in custom_resNet_model.layers[:-1]:
        layer.trainable = False

    #custom_resNet_model.summary()
    custom_resNet_model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    t = time.time()
    custom_resNet_model.fit(trainData, trainLabels, batch_size=32, epochs=10, verbose=0)
    print('Training time: %s' % (time.time()-t))

    return custom_resNet_model



def Test(model, testData):
    """
    Implementing our trained classifying model over the test data
    :param model: the trained model
    :param testData: array of the test data
    :return: the predictions (the CNN score - probability to be a flower) for each test sample
    """
    results = model.predict(testData)
    return results


def Evaluate(results, splitData):
    testData, testLabels = splitData['TestData'], SplitData['TestLabels']
    predictions = results > 0.5
    summ = np.sum(splitData['TrainLabels'])
    sum = 0
    for i in range(0, len(predictions)):
        if predictions[i] == testLabels[i]:
            sum = sum + 1
    errorRate = 1-(sum/len(predictions))
    type1Errors = {'Indices': [], 'Results': []}
    type2Errors = {'Indices': [], 'Results': []}
    for i in range(0, len(predictions)):
        imageIndex = test_images_indices[i]
        if (predictions[i] == 0) and (testLabels[i] == 1):
            type1Errors['Indices'].append(imageIndex)
            type1Errors['Results'].append(results[i])
        if (predictions[i] == 1) and (testLabels[i] == 0):
            type2Errors['Indices'].append(imageIndex)
            type2Errors['Results'].append(results[i])
    print(type1Errors['Indices'])
    print(type2Errors['Indices'])
    topType1Errors = {'Indices': [], 'Results': []}
    topType2Errors = {'Indices': [], 'Results': []}
    for k in range(0,5):
        if len(type1Errors['Indices']) > 0:
            localMinIndex = type1Errors['Results'].index(min(type1Errors['Results']))
            globalMinIndex = type1Errors['Indices'][localMinIndex]
            MinError = type1Errors['Results'][localMinIndex]
            topType1Errors['Indices'].append(globalMinIndex)
            topType1Errors['Results'].append(MinError)

            type1Errors['Results'].pop(localMinIndex)
            type1Errors['Indices'].pop(localMinIndex)
        if len(type2Errors['Indices']) > 0:
            localMinIndex = type2Errors['Results'].index(max(type2Errors['Results']))
            globalMinIndex = type2Errors['Indices'][localMinIndex]
            MinError = type2Errors['Results'][localMinIndex]
            topType2Errors['Indices'].append(globalMinIndex)
            topType2Errors['Results'].append(MinError)

            type2Errors['Results'].pop(localMinIndex)
            type2Errors['Indices'].pop(localMinIndex)
    print(topType1Errors['Indices'])
    print(topType2Errors['Indices'])
    return {'ErrorRate': errorRate, 'Type1': topType1Errors, 'Type2': topType2Errors}





def ReportResults(Summary, Results):
    print("Test Error: " + str(Summary['ErrorRate']))
    for i in range(0, len(Summary['Type1']['Indices'])):
        index = Summary['Type1']['Indices'][i]
        img = cv2.imread(data_path + '/' + str(index) + '.jpeg')
        plt.imshow(img)
        plt.title("Error Type 1, #" + str(i+1) + " ,Image Name: " + str(index) + ", Classification Score: "
                  + str(Summary['Type1']['Results'][i]))
        plt.show()
    for i in range(0, len(Summary['Type2']['Indices'])):
        index = Summary['Type2']['Indices'][i]
        img = cv2.imread(data_path + '/' + str(i+1) + '.jpeg')
        plt.imshow(img)
        plt.title("Error Type 2, #" + str(i+1) + " ,Image Name: " + str(index) + ", Classification Score: "
                  + str(Summary['Type2']['Results'][i]))
        plt.show()

    testLabels= []
    for i in range(0, len(SplitData['TestLabels'])):
        testLabels.append(SplitData['TestLabels'][i][0])
    scores = []
    for i in range(0, len(Results)):
        scores.append(Results[i][0])
    print(scores)
    precision, recall, thresholds = precision_recall_curve(testLabels, scores)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()


#----------Main_Improve2------------------

data_path = "C:/Users/wr/Downloads/FlowerData/"
test_images_indices = list(range(301, 473))
Params = getParameters()
DandL = InitData(data_path, Params)
SplitData = TrainTestSplit(DandL['Data'], DandL['Labels'])
Model = Train(SplitData)
#pickle.dump(Model, open( "save_improve2_model.p", "wb" ))
#Model = pickle.load( open( "save_improve2_model.p", "rb" ) )
#history = fitModel(Model, SplitData['TrainData'], SplitData['TrainLabels'])
Results = Test(Model, SplitData['TestData'])
#pickle.dump(Results, open( "save_improve2_results.p", "wb" ))
#Results = pickle.load( open( "save_improve2_results.p", "rb" ) )
Summary = Evaluate(Results, SplitData)
ReportResults(Summary, Results)

# ------ Improvement 3: Increase Learned Layers-----------------------------------

def getParameters():
    s = 224
    params = {'S': s}
    return params


def InitData(path, params):
    """
    loads the data from the desired path
    :param path: the path of the data folders
    :param  params: dictionary of all the parameters
    :return: dictionary of 2 lists- Data (images) and the Labels (classes)
    """
    DandL = {'Data': [], 'Labels': []}
    dirsStrings = os.listdir(path)
    dirs = []
    for i in range(0, len(dirsStrings)-1):
        dirs.append(int(dirsStrings[i][:-5]))
    dirs = sorted(dirs)
    DandL['Labels'] = sio.loadmat(path + 'FlowerDataLabels.mat')
    DandL['Labels'] = DandL['Labels']['Labels']
    DandL['Labels'] = DandL['Labels'].reshape(DandL['Labels'].size, 1)
    s = params['S']
    for tempImageNum in dirs:
        img_path = path + '/' + str(tempImageNum) + '.jpeg'
        #img = cv2.imread(img_path)
        #img = cv2.resize(img, (s, s), interpolation=cv2.INTER_CUBIC)
        img = image.load_img(img_path, target_size=(s, s))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x.astype(float)
        x = preprocess_input(x)
        DandL['Data'].append(x)
    DandL['Data'] = np.array(DandL['Data'])
    DandL['Data'] = np.rollaxis(DandL['Data'], 1, 0)
    DandL['Data'] = DandL['Data'][0]
    return DandL


def TrainTestSplit(data, labels):
    """
    splits the data into train data and test data with their respective labels
    :param data: all the data (images) that has been loaded
    :param labels: all the labels of the data (classes)
    :return: dictionary of 4 lists- 2 lists for the train (data and labels) and 2 for the test
    """
    splitData = {'TrainData': [], 'TrainLabels': [], 'TestData': [], 'TestLabels': []}
    all_images_indices = range(1,473)
    train_images_indices = list(set(all_images_indices) - set(test_images_indices))
    zeroValueTrain_images_indices = [x - 1 for x in train_images_indices]
    zeroValueTest_images_indices = [x - 1 for x in test_images_indices]
    splitData['TrainData'] = data[zeroValueTrain_images_indices]
    splitData['TestData'] = data[zeroValueTest_images_indices]
    splitData['TrainLabels'] = labels[zeroValueTrain_images_indices]
    splitData['TestLabels'] = labels[zeroValueTest_images_indices]
    splitData['TrainData'] = np.asarray(splitData['TrainData'])
    splitData['TestData'] = np.asarray(splitData['TestData'])
    splitData['TrainLabels'] = np.asarray(splitData['TrainLabels'])
    splitData['TestLabels'] = np.asarray(splitData['TestLabels'])

    return splitData


def Train(SplitData):
    """
    Train the last layer of the network over the train data
    :param SplitData: a dict of the data (images+labels) splits to train and test, will get the value returned from TrainTestSplit function
    :return: the trained model
    """
    trainData, trainLabels = SplitData['TrainData'], SplitData['TrainLabels']

    tensor_input = Input(shape=(224, 224, 3))
    model = keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet', pooling= 'avg')
    #model.summary()

    our_last_layer = model.get_layer('avg_pool').output
    out = Dense(1, activation='sigmoid', name='output')(our_last_layer)
    custom_resNet_model = Model(model.input, out)


    # freeze the layers so we don't train them
    for layer in custom_resNet_model.layers[:-9]:
        layer.trainable = False

    #custom_resNet_model.summary()
    custom_resNet_model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    t = time.time()
    custom_resNet_model.fit(trainData, trainLabels, batch_size=32, epochs=10, verbose=0)
    print('Training time: %s' % (time.time()-t))

    return custom_resNet_model



def Test(model, testData):
    """
    Implementing our trained classifying model over the test data
    :param model: the trained model
    :param testData: array of the test data
    :return: the predictions (the CNN score - probability to be a flower) for each test sample
    """
    results = model.predict(testData)
    return results


def Evaluate(results, splitData):
    testData, testLabels = splitData['TestData'], SplitData['TestLabels']
    predictions = results > 0.5
    summ = np.sum(splitData['TrainLabels'])
    sum = 0
    for i in range(0, len(predictions)):
        if predictions[i] == testLabels[i]:
            sum = sum + 1
    errorRate = 1-(sum/len(predictions))
    type1Errors = {'Indices': [], 'Results': []}
    type2Errors = {'Indices': [], 'Results': []}
    for i in range(0, len(predictions)):
        imageIndex = test_images_indices[i]
        if (predictions[i] == 0) and (testLabels[i] == 1):
            type1Errors['Indices'].append(imageIndex)
            type1Errors['Results'].append(results[i])
        if (predictions[i] == 1) and (testLabels[i] == 0):
            type2Errors['Indices'].append(imageIndex)
            type2Errors['Results'].append(results[i])
    print(type1Errors['Indices'])
    print(type2Errors['Indices'])
    topType1Errors = {'Indices': [], 'Results': []}
    topType2Errors = {'Indices': [], 'Results': []}
    for k in range(0,5):
        if len(type1Errors['Indices']) > 0:
            localMinIndex = type1Errors['Results'].index(min(type1Errors['Results']))
            globalMinIndex = type1Errors['Indices'][localMinIndex]
            MinError = type1Errors['Results'][localMinIndex]
            topType1Errors['Indices'].append(globalMinIndex)
            topType1Errors['Results'].append(MinError)

            type1Errors['Results'].pop(localMinIndex)
            type1Errors['Indices'].pop(localMinIndex)
        if len(type2Errors['Indices']) > 0:
            localMinIndex = type2Errors['Results'].index(max(type2Errors['Results']))
            globalMinIndex = type2Errors['Indices'][localMinIndex]
            MinError = type2Errors['Results'][localMinIndex]
            topType2Errors['Indices'].append(globalMinIndex)
            topType2Errors['Results'].append(MinError)

            type2Errors['Results'].pop(localMinIndex)
            type2Errors['Indices'].pop(localMinIndex)
    print(topType1Errors['Indices'])
    print(topType2Errors['Indices'])
    return {'ErrorRate': errorRate, 'Type1': topType1Errors, 'Type2': topType2Errors}





def ReportResults(Summary, Results):
    print("Test Error: " + str(Summary['ErrorRate']))
    for i in range(0, len(Summary['Type1']['Indices'])):
        index = Summary['Type1']['Indices'][i]
        img = cv2.imread(data_path + '/' + str(index) + '.jpeg')
        plt.imshow(img)
        plt.title("Error Type 1, Error Index:  " + str(i+1) + ", Classification Score: "
                  + str(Summary['Type1']['Results'][i]))
        plt.show()
    for i in range(0, len(Summary['Type2']['Indices'])):
        index = Summary['Type2']['Indices'][i]
        img = cv2.imread(data_path + '/' + str(i+1) + '.jpeg')
        plt.imshow(img)
        plt.title("Error Type 2, Error Index:  " + str(index) + ", Classification Score: "
                  + str(Summary['Type2']['Results'][i]))
        plt.show()

    testLabels= []
    for i in range(0, len(SplitData['TestLabels'])):
        testLabels.append(SplitData['TestLabels'][i][0])
    scores = []
    for i in range(0, len(Results)):
        scores.append(Results[i][0])
    print(scores)
    precision, recall, thresholds = precision_recall_curve(testLabels, scores)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()


#----------Main_Improve3------------------

data_path = "C:/Users/wr/Downloads/FlowerData/"
test_images_indices = list(range(301, 473))
Params = getParameters()
DandL = InitData(data_path, Params)
SplitData = TrainTestSplit(DandL['Data'], DandL['Labels'])
Model = Train(SplitData)
#pickle.dump(Model, open( "save_improve3_model.p", "wb" ))
#Model = pickle.load( open( "save_improve3_model.p", "rb" ) )
#history = fitModel(Model, SplitData['TrainData'], SplitData['TrainLabels'])
Results = Test(Model, SplitData['TestData'])
#pickle.dump(Results, open( "save_improve3_results.p", "wb" ))
#Results = pickle.load( open( "save_improve3_results.p", "rb" ) )
Summary = Evaluate(Results, SplitData)
ReportResults(Summary, Results)

# ------ Improvement 4: Tuning Parameters-----------------------------------------

def getParameters():
    s = 224
    params = {'S': s}
    return params


def InitData(path, params):
    """
    loads the data from the desired path
    :param path: the path of the data folders
    :param  params: dictionary of all the parameters
    :return: dictionary of 2 lists- Data (images) and the Labels (classes)
    """
    DandL = {'Data': [], 'Labels': []}
    dirsStrings = os.listdir(path)
    dirs = []
    for i in range(0, len(dirsStrings)-1):
        dirs.append(int(dirsStrings[i][:-5]))
    dirs = sorted(dirs)
    DandL['Labels'] = sio.loadmat(path + 'FlowerDataLabels.mat')
    DandL['Labels'] = DandL['Labels']['Labels']
    DandL['Labels'] = DandL['Labels'].reshape(DandL['Labels'].size, 1)
    s = params['S']
    for tempImageNum in dirs:
        img_path = path + '/' + str(tempImageNum) + '.jpeg'
        #img = cv2.imread(img_path)
        #img = cv2.resize(img, (s, s), interpolation=cv2.INTER_CUBIC)
        img = image.load_img(img_path, target_size=(s, s))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x.astype(float)
        x = preprocess_input(x)
        DandL['Data'].append(x)
    DandL['Data'] = np.array(DandL['Data'])
    DandL['Data'] = np.rollaxis(DandL['Data'], 1, 0)
    DandL['Data'] = DandL['Data'][0]
    return DandL


def TrainTestSplit(data, labels):
    """
    splits the data into train data and test data with their respective labels
    :param data: all the data (images) that has been loaded
    :param labels: all the labels of the data (classes)
    :return: dictionary of 4 lists- 2 lists for the train (data and labels) and 2 for the test
    """
    splitData = {'TrainData': [], 'TrainLabels': [], 'TestData': [], 'TestLabels': []}
    all_images_indices = range(1,473)
    train_images_indices = list(set(all_images_indices) - set(test_images_indices))
    zeroValueTrain_images_indices = [x - 1 for x in train_images_indices]
    zeroValueTest_images_indices = [x - 1 for x in test_images_indices]
    splitData['TrainData'] = data[zeroValueTrain_images_indices]
    splitData['TestData'] = data[zeroValueTest_images_indices]
    splitData['TrainLabels'] = labels[zeroValueTrain_images_indices]
    splitData['TestLabels'] = labels[zeroValueTest_images_indices]
    splitData['TrainData'] = np.asarray(splitData['TrainData'])
    splitData['TestData'] = np.asarray(splitData['TestData'])
    splitData['TrainLabels'] = np.asarray(splitData['TrainLabels'])
    splitData['TestLabels'] = np.asarray(splitData['TestLabels'])

    return splitData


def Train(SplitData, params):
    """
    Train the last layer of the network over the train data
    :param SplitData: a dict of the data (images+labels) splits to train and test, will get the value returned from TrainTestSplit function
    :return: the trained model
    """
    trainData, trainLabels = SplitData['TrainData'], SplitData['TrainLabels']

    tensor_input = Input(shape=(224, 224, 3))
    model = keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet', pooling= 'avg')
    #model.summary()

    our_last_layer = model.get_layer('avg_pool').output
    out = Dense(1, activation='sigmoid', name='output')(our_last_layer)
    custom_resNet_model = Model(model.input, out)

    # freeze the layers so we don't train them
    for layer in custom_resNet_model.layers[:-1]:
        layer.trainable = False

    #custom_resNet_model.summary()
    opt = optimizers.SGD(lr=params['LR'], decay=params['Decay'], momentum=params['Momentum'])
    custom_resNet_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    #t = time.time()
    custom_resNet_model.fit(trainData, trainLabels, batch_size=32, epochs=10, verbose=0)
    #print('Training time: %s' % (time.time()-t))

    return custom_resNet_model



def Test(model, testData):
    """
    Implementing our trained classifying model over the test data
    :param model: the trained model
    :param testData: array of the test data
    :return: the predictions (the CNN score - probability to be a flower) for each test sample
    """
    results = model.predict(testData)
    return results


def Evaluate(results, splitData, params):
    testData, testLabels = splitData['TestData'], SplitData['TestLabels']
    predictions = results > params['T']
    sum = 0
    for i in range(0, len(predictions)):
        if predictions[i] == testLabels[i]:
            sum = sum + 1
    errorRate = 1-(sum/len(predictions))
    type1Errors = {'Indices': [], 'Results': []}
    type2Errors = {'Indices': [], 'Results': []}
    for i in range(0, len(predictions)):
        imageIndex = test_images_indices[i]
        if (predictions[i] == 0) and (testLabels[i] == 1):
            type1Errors['Indices'].append(imageIndex)
            type1Errors['Results'].append(results[i])
        if (predictions[i] == 1) and (testLabels[i] == 0):
            type2Errors['Indices'].append(imageIndex)
            type2Errors['Results'].append(results[i])
    print(type1Errors['Indices'])
    print(type2Errors['Indices'])
    topType1Errors = {'Indices': [], 'Results': []}
    topType2Errors = {'Indices': [], 'Results': []}
    for k in range(0,5):
        if len(type1Errors['Indices']) > 0:
            localMinIndex = type1Errors['Results'].index(min(type1Errors['Results']))
            globalMinIndex = type1Errors['Indices'][localMinIndex]
            MinError = type1Errors['Results'][localMinIndex]
            topType1Errors['Indices'].append(globalMinIndex)
            topType1Errors['Results'].append(MinError)

            type1Errors['Results'].pop(localMinIndex)
            type1Errors['Indices'].pop(localMinIndex)
        if len(type2Errors['Indices']) > 0:
            localMinIndex = type2Errors['Results'].index(max(type2Errors['Results']))
            globalMinIndex = type2Errors['Indices'][localMinIndex]
            MinError = type2Errors['Results'][localMinIndex]
            topType2Errors['Indices'].append(globalMinIndex)
            topType2Errors['Results'].append(MinError)

            type2Errors['Results'].pop(localMinIndex)
            type2Errors['Indices'].pop(localMinIndex)
    print(topType1Errors['Indices'])
    print(topType2Errors['Indices'])
    return {'ErrorRate': errorRate, 'Type1': topType1Errors, 'Type2': topType2Errors}





def ReportResults(Summary, Results):
    print("Test Error: " + str(Summary['ErrorRate']))
    for i in range(0, len(Summary['Type1']['Indices'])):
        index = Summary['Type1']['Indices'][i]
        img = cv2.imread(data_path + '/' + str(index) + '.jpeg')
        plt.imshow(img)
        plt.title("Error Type 1, Error Index:  " + str(i+1) + ", Classification Score: "
                  + str(Summary['Type1']['Results'][i]))
        plt.show()
    for i in range(0, len(Summary['Type2']['Indices'])):
        index = Summary['Type2']['Indices'][i]
        img = cv2.imread(data_path + '/' + str(i+1) + '.jpeg')
        plt.imshow(img)
        plt.title("Error Type 2, Error Index:  " + str(index) + ", Classification Score: "
                  + str(Summary['Type2']['Results'][i]))
        plt.show()

    testLabels= []
    for i in range(0, len(SplitData['TestLabels'])):
        testLabels.append(SplitData['TestLabels'][i][0])
    scores = []
    for i in range(0, len(Results)):
        scores.append(Results[i][0])
    print(scores)
    precision, recall, thresholds = precision_recall_curve(testLabels, scores)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()


def EqualsDicts(dict1, dict2):
    flag = True
    if dict1['T'] != dict2['T']:
        flag = False
    if dict1['LR'] != dict2['LR']:
        flag = False
    if dict1['Decay'] != dict2['Decay']:
        flag = False
    if dict1['Momentum'] != dict2['Momentum']:
        flag = False
    return flag


def ValidationEvaluate(results, tuningData, params):
    valData, valLabels = tuningData['ValData'], tuningData['ValLabels']
    predictions = results > params['T']
    sum = 0
    for i in range(0, len(predictions)):
        if predictions[i] == valLabels[i]:
            sum = sum + 1
    errorRate = 1 - (sum / len(predictions))
    return errorRate


def findValError(tuningData, params):
    model = Train(tuningData, params)
    results = Test(model, tuningData['ValData'])
    valError = ValidationEvaluate(results, tuningData, params)
    return valError


def PrintPlots(tValuesAndErrors, lrValuesAndErrors, decayValuesAndErrors, momentumValuesAndErrors):
    plt.plot(tValuesAndErrors['Values'], tValuesAndErrors['Errors'])
    plt.xlabel('T values')
    plt.ylabel('Validation Error')
    plt.savefig('T.png')
    plt.close()

    plt.semilogx(lrValuesAndErrors['Values'], lrValuesAndErrors['Errors'])
    plt.xlabel('LR values')
    plt.ylabel('Validation Error')
    plt.savefig('LR_log.png')
    plt.close()

    plt.plot(lrValuesAndErrors['Values'], lrValuesAndErrors['Errors'])
    plt.xlabel('LR values')
    plt.ylabel('Validation Error')
    plt.savefig('LR.png')
    plt.close()

    plt.semilogx(decayValuesAndErrors['Values'], decayValuesAndErrors['Errors'])
    plt.xlabel('Decay values')
    plt.ylabel('Validation Error')
    plt.savefig('Decay_log.png')
    plt.close()

    plt.plot(decayValuesAndErrors['Values'], decayValuesAndErrors['Errors'])
    plt.xlabel('Decay values')
    plt.ylabel('Validation Error')
    plt.savefig('Decay.png')
    plt.close()

    plt.plot(momentumValuesAndErrors['Values'], momentumValuesAndErrors['Errors'])
    plt.xlabel('Momentum values')
    plt.ylabel('Validation Error')
    plt.savefig(str(type) + ' Momentum.png')
    plt.close()


def Tuning(trainData, trainLabels):
    TuningData = {'TrainData': [], 'TrainLabels': [], 'ValData': [], 'ValLabels': []}
    splitIndex = round(0.8*len(trainData))
    TuningData['TrainData'] = trainData[0:splitIndex]
    TuningData['TrainLabels'] = trainLabels[0:splitIndex]
    TuningData['ValData'] = trainData[splitIndex:]
    TuningData['ValLabels'] = trainLabels[splitIndex:]
    s = 224
    params = {'S': s, 'T': 0.5, 'LR': 0.01, 'Decay': 0.01, 'Momentum': 0.5}
    currentParams = copy.deepcopy(params)
    print("Initial:")
    print("Current Params = " + str(currentParams))
    for iterarion in range(0, 5):
        previousParams = copy.deepcopy(currentParams)
        print("Start Iteration:")
        print("Previous Params = " + str(previousParams))
        print("Current Params = " + str(currentParams))
        tValuesAndErrors = {'Values': np.arange(0.5, 0.91, 0.1), 'Errors': []}
        for t in tValuesAndErrors['Values']:
            currentParams['T'] = t
            ValError = findValError(TuningData, currentParams)
            tValuesAndErrors['Errors'].append(ValError)
        minErrorIndex = tValuesAndErrors['Errors'].index(min(tValuesAndErrors['Errors']))
        sMinError = tValuesAndErrors['Values'][minErrorIndex]
        currentParams['T'] = sMinError
        print("best t for type " + str(type) + " is " + str(currentParams['T']))

        lrValuesAndErrors = {'Values': np.logspace(-4, 0, 5), 'Errors': []}
        for lr in lrValuesAndErrors['Values']:
            currentParams['LR'] = lr
            ValError = findValError(TuningData, currentParams)
            lrValuesAndErrors['Errors'].append(ValError)
        minErrorIndex = lrValuesAndErrors['Errors'].index(min(lrValuesAndErrors['Errors']))
        lrMinError = lrValuesAndErrors['Values'][minErrorIndex]
        currentParams['LR'] = lrMinError
        print("best lr for type " + str(type) + " is " + str(currentParams['LR']))

        decayValuesAndErrors = {'Values': [0, 0.001, 0.01, 0.1, 1], 'Errors': []}
        for decay in decayValuesAndErrors['Values']:
            currentParams['Decay'] = decay
            ValError = findValError(TuningData, currentParams)
            decayValuesAndErrors['Errors'].append(ValError)
        minErrorIndex = decayValuesAndErrors['Errors'].index(min(decayValuesAndErrors['Errors']))
        decayMinError = decayValuesAndErrors['Values'][minErrorIndex]
        currentParams['Decay'] = decayMinError
        print("best decay for type " + str(type) + " is " + str(currentParams['Decay']))

        momentumValuesAndErrors = {'Values': [0, 0.1, 0.3, 0.5, 0.7, 0.9], 'Errors': []}
        for momentum in momentumValuesAndErrors['Values']:
            currentParams['Momentum'] = momentum
            ValError = findValError(TuningData, currentParams)
            momentumValuesAndErrors['Errors'].append(ValError)
        minErrorIndex = momentumValuesAndErrors['Errors'].index(min(momentumValuesAndErrors['Errors']))
        momentumMinError = momentumValuesAndErrors['Values'][minErrorIndex]
        currentParams['Momentum'] = momentumMinError
        print("best momentum for type " + str(type) + " is " + str(currentParams['Momentum']))

        if EqualsDicts(currentParams, previousParams):
            break
    PrintPlots(tValuesAndErrors, lrValuesAndErrors, decayValuesAndErrors, momentumValuesAndErrors)
    return currentParams


#----------Main_Improve4------------------

data_path = "C:/Users/wr/Downloads/FlowerData/"
test_images_indices = list(range(301, 473))
Params = getParameters()
DandL = InitData(data_path, Params)
SplitData = TrainTestSplit(DandL['Data'], DandL['Labels'])
Params = Tuning(SplitData['TrainData'], SplitData['TrainLabels'])
Model = Train(SplitData, Params)
#pickle.dump(Model, open( "save_improve4_model.p", "wb" ))
#Model = pickle.load( open( "save_improve4_model.p", "rb" ) )
#history = fitModel(Model, SplitData['TrainData'], SplitData['TrainLabels'])
Results = Test(Model, SplitData['TestData'])
#pickle.dump(Results, open( "save_improve4_results.p", "wb" ))
#Results = pickle.load( open( "save_improve4_results.p", "rb" ) )
Summary = Evaluate(Results, SplitData, Params)
ReportResults(Summary, Results)

'''''
