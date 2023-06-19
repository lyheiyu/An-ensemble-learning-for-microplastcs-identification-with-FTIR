import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import math
class FTIRByPLS:
    def __init__(self):
        pass

    def parseData2(fileName, begin, end):

        dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False,low_memory=False)
        polymerName = dataset.iloc[1:971, 1]
        waveLength = dataset.iloc[0, begin:end]
        intensity = dataset.iloc[1:971, begin:end]
        polymerName = np.array(polymerName)
        waveLength = np.array(waveLength)
        # polymerNameList = {}.fromkeys(polymerName).keys()
        polymerNameList = []
        pList = list(set(polymerName))
        for item in pList:
            #print(item)
            polymerNameList.append(item)
        polymerNameList = np.array(polymerNameList)
        #print(polymerNameList.shape)
        polymerNameID = []
        for i in range(len(polymerNameList)):
            polymerNameID.append(i + 1)
        #print(polymerNameID)
        polymerNameData = []
        for item in polymerName:
            for i in range(len(pList)):
                if item == pList[i]:
                    polymerNameData.append(i + 1)
        #print(polymerNameData)

        intensity = np.array(intensity)
        return polymerName, waveLength, intensity, polymerNameData
    def spectrumPls(x,y,number):

        pls = PLSRegression(n_components=number)
        pls.fit(x, y)

        return pls
    def getAccuracy(x_train,y_train,x_test,y_test,number):

        accuracies = []
        #index = []
        model =FTIRByPLS.spectrumPls(x_train, y_train,number)
        #index.append(i)
        index=[]
        for i in range(len(x_test)):
            index.append(i+1)
        y_predict = model.predict(x_test).flatten()
        fig, ax = plt.subplots()
        ax.set_xlabel('i')
        ax.set_ylabel('accuracy')
        ax.plot(index, y_predict, 'r', label='Pls', linestyle='dotted')
        ax.plot(index, y_test, 'b', label='realPLs', linestyle='dotted')
        ax.legend()
        plt.show()
        #print(y_predict)
        sum = len(y_predict)
        correct = 0
        for j in range(len(y_predict)):
            # print(y_predict[j])
            # print(y_test[j])
            if y_predict[j] == y_test[j]:

                correct = correct + 1
        accuracies.append(correct / sum)

        # print(index)

        print('accuracy:' + str(correct / sum))
        return correct / sum

    def PLS(fileName):
        for tag in [5,6,7,8,9,10,11,12,13]:
            #x, y = data_pre.load('data_test_mean_mean.csv')
            polymerName, waveLength, intensity, polymerID = tR.parseData2(fileName, 2, 1763)
            intensity=normalize(intensity,norm='l2')
            x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.2, random_state=1)
            pls2 = PLSRegression(n_components=tag)
            x_train=np.array(x_train)
            x_test=np.array(x_test)
            y_train=np.array(y_train)
            y_test=np.array(y_test)
            pls2.fit(x_train, y_train.ravel())
            y_test_pre = pls2.predict(x_test).flatten()
            sumd, mxd, cnt = 0.0, 0.0, 0.0
            for i in range(len(y_test_pre)):
                if (math.fabs(y_test_pre[i] - y_test[i]) > 0.5):
                    cnt += 1
                sumd += math.fabs(y_test_pre[i] - y_test[i])
                mxd = max(mxd, math.fabs(y_test_pre[i] - y_test[i]))
            sumd /= len(y_test_pre)
            cnt /= len(y_test_pre)
            accuracy = pls2.score(x_test, y_test)
            plt.xlabel('components')
            plt.ylabel('accuracy')

            plt.scatter(tag,accuracy,label='PLS')
            print(str(tag)+':'+'{:.2f},{:.2f},{:.2f},{:.2f}%'.format(accuracy, sumd, mxd, cnt * 100))
        plt.show()
if __name__ == '__main__':
    tR=FTIRByPLS
    polymerName, waveLength, intensity, polymerID = tR.parseData2('D4_4_publication.csv', 2, 1763)
    intensity = normalize(intensity, norm='l2')
    x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.3, random_state=1)

    index = []
    acc = []
    # for i in range(1):
    #     index.append(i + 1)
    #acc.append(tR.getAccuracy(x_train, y_train, x_test, y_test,10))
    #print(max(acc), min(acc), sum(acc))
    tR.PLS('D4_4_publication.csv')