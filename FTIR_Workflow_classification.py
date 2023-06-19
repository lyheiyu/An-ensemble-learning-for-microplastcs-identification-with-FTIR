import pandas as pd
import numpy as np
from time import time
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import cohen_kappa_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm
from plsda import PLSDA
from simca import SIMCA
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from FTIR_KNN import TestKNN
scores = [cohen_kappa_score,f1_score, accuracy_score, recall_score,precision_score]

def meanX(dataX):
    return np.mean(dataX, axis=0)  # axis=0表示按照列来求均值，如果输入list,则axis=1


def pca(XMat, k):
    average = meanX(XMat)

    m, n = np.shape(XMat)
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T)  # 计算协方差矩阵
    featValue, featVec = np.linalg.eig(covX)  # 求解协方差矩阵的特征值和特征向量
    index = np.argsort(-featValue)  # 按照featValue进行从大到小排序
    finalData = []
    if k > n:
        print("k must lower than feature number")
        return
    else:
        # 注意特征向量时列向量
        selectVec = np.matrix(featVec.T[index[:k]])  # 所以这里需要进行转置，而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
        print('############', selectVec[0].reshape(1, -1))
        PC1 = selectVec[0].reshape(1761)
        print(PC1.shape)
        finalData = data_adjust * selectVec.T
        reconData = (finalData * selectVec) + average
    return finalData, reconData, selectVec


def parseData2(fileName, begin, end):
    dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False, low_memory=False)

    polymerName = dataset.iloc[1:, 1]
    waveLength = dataset.iloc[0, begin:end]
    intensity = dataset.iloc[1:, begin:end]
    polymerName = np.array(polymerName)
    # waveLength = np.array(waveLength)

    polymerID = dataset.iloc[1:, 1764]
    polymerID = np.array(polymerID)
    intensity = np.array(intensity)
    x_class = []
    y_class = []
    for i in range(2):
        m = []
        z = []
        for j in range(len(intensity)):
            if int(polymerID[j]) == i + 1:
                z.append(polymerID[j])
                m.append(intensity[j])
        x_class.append(m)
        y_class.append(z)

    return polymerName, waveLength, intensity, polymerID, x_class, y_class


polymerName, waveLength, intensity, polymerID = TestKNN.parseData2('D4_4_publication_svm.csv', 2, 1763)


def plotBestFit(data1, data2):
    dataArr1 = np.array(data1)
    # dataArr2 = np.array(data2)

    m = np.shape(dataArr1)[0]
    axis_x1 = []
    axis_y1 = []
    axis_x2 = []
    axis_y2 = []
    for i in range(m):
        axis_x1.append(dataArr1[i, 0])
        axis_y1.append(dataArr1[i, 1])
        # axis_x2.append(dataArr2[i,0])
        # axis_y2.append(dataArr2[i,1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(axis_x1, axis_y1, s=50, c='red', marker='s')
    # ax.scatter(axis_x2, axis_y2, s=50, c='blue')
    plt.xlabel('x1');
    plt.ylabel('x2');
    # plt.savefig("outfile.png")
    plt.show()


def main(i):
    datafile = "data.txt"
    polymerName, waveLength, intensity, polymerID, x_class, y_class = parseData2('D4_4_publication5.csv', 2, 1763)

    k = (i + 1)
    return pca(intensity, k)


if __name__ == '__main__':
    finalData, reconMat, Vector = main(120)

    finalData = np.array(finalData, dtype=np.float32)
    print(finalData.shape)
    # ohe = sp.OneHotEncoder(sparse=False, dtype="int32")
    # result = ohe.fit_transform(finalData)
    # finalData = MMScaler.fit_transform(finalData)
    # modelMLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128,128), random_state=1)
    polymerName, waveLength, intensity, polymerID, x_class, y_class = parseData2('D4_4_publication_svm.csv', 2, 1763)
    x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.25, random_state=1)
    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    modelSVM = svm.SVC(C=1, kernel='linear', decision_function_shape='ovo')
    modelSVM.fit(x_train,y_train)
    reg=[modelSVM]
    y_pred=modelSVM.predict(x_test)
    correct=0
    for i in range(len(y_pred)):
        if y_test[i]==y_pred[i]:
            correct+=1
        print(y_test[i],y_pred[i])

    print(correct/len(y_pred))
    thisscore=[]
    for score in scores:
        thisscore.append(score(y_test, y_pred))
        print(reg.__class__.__name__ + '---' + score.__name__ + ':', score(y_test, y_pred))
