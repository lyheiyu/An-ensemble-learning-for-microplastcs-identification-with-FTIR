import random

import numpy
import pandas as pd
import numpy as np
import keras_metrics as km
from time import time
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import cohen_kappa_score
from joblib import dump, load
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm
# from plsda import PLSDA
# from simca import SIMCA
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from FTIR_KNN import TestKNN
from PLS import airPLS
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report
scores = [cohen_kappa_score,f1_score, accuracy_score, recall_score,precision_score]
from plsda import PLSDA
from sklearn.ensemble import VotingClassifier
def meanX(dataX):
    return np.mean(dataX, axis=0)  # axis=0表示按照列来求均值，如果输入list,则axis=1

def getPN(fileName):
    dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False, low_memory=False)

    polymerName = dataset.iloc[1:, 1]
    PN = []
    for item in polymerName:
        if item not in PN:
            PN.append(item)
    return PN
def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    plt.imshow(cm, interpolation='nearest')  # 在特定的窗口上显示图像
    plt.title(title)  # 图像标题
    plt.colorbar()
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 30,
             }
    plt.title(title,font2)  # 图像标题
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90, size=20)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name, size=20)  # 将标签印在y轴坐标上
    plt.ylabel('True label', font2)
    plt.xlabel('Predicted label', font2)
    plt.show()
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
    intensity = np.array(intensity, dtype=np.float32)
    intensityBaseline = []
    for item in intensity:
        item = item - airPLS(item)
        intensityBaseline.append(item)

    # intensity=intensity-airPLS(intensity)

    # scaler= StandardScaler().fit(intensityBaseline)
    # intensity=scaler.transform(intensityBaseline)
    # minScaler = MinMaxScaler().fit(intensityBaseline)
    # intensity= minScaler.transform(intensityBaseline)
    # l1<l2<max max performs the best of this three type nor
    intensity = normalize(intensityBaseline, 'max')
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

from utils import utils
polymerName, waveLength, intensity, polymerID,x_each,y_each = utils.parseData2('D4_4_publication5.csv', 2, 1763)


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

    # this is for the 216_2018.csv
    # polymerName, polymerID, waveLength, intensity=tk.parseData2('216_2018_1156_MOESM5_ESM.csv',3,1179)
    # this is for the D4 public_csv
    # kf=KFold(n_splits=3)


    polymerName, waveLength, intensity, polymerID, x_each, y_each = utils.parseData2('D4_4_publication5.csv', 2,
                                                                                     1763)
    # MLPmodel = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(64, 64, 64), random_state=1)
    KNNModel = KNeighborsClassifier(n_neighbors=2)
    KNNModelPCA = KNeighborsClassifier(n_neighbors=2)
    SVM_clf = svm.SVC(C=0.3, kernel='linear', decision_function_shape='ovo',probability=True)
    SVM_clfPCA = svm.SVC(C=0.3, kernel='linear', decision_function_shape='ovo',probability=True)
    RandomF = RandomForestClassifier(n_estimators=74, oob_score=True,
                                     random_state=10, max_depth=11,
                                     min_samples_split=2, min_samples_leaf=2, max_features=9)
    LDA_clf_PCA = LinearDiscriminantAnalysis(n_components=10, priors=None, shrinkage=None,
                                         solver='svd', store_covariance=True, tol=0.0001)
    sp = PLSDA(n_components=30, style='hard')

    cmTotal = np.zeros((12, 12))

    PCAModel = load('model/pca200.joblib')
    #models = [SVM_clf, MLPmodel, KNNModel, RandomF, LDA_clf, sp]
    PCAintensity=PCAModel.transform(intensity)
    totalReport = []
    PN = getPN('D4_4_publication5.csv')
    maxnumber=[]
    for p in range(1):
        m = 0
        t_report = []
        scoreTotal = np.zeros(5)

        for i in range(200):
            totalProbability = []
            totalProbabilityForTest = []
            polymerName, waveLength, intensity, polymerID, x_each, y_each = utils.parseData2('D4_4_publication5.csv', 2,
                                                                                             1763)
            preName = []
            preID = []
            x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.3, random_state=i)
            x_train_PCA, x_test_PCA, y_train_PCA, y_test_PCA = train_test_split(PCAintensity, polymerID, test_size=0.3, random_state=i)


            test = set(y_train.tolist())
            print(len(test))
            if(len(test)==11):
                test=[]
                continue
            # model = tS.spectrumSVM(x_train, y_train, 0.3, 'linear', 'ovo')
            # KNNModelPCA.fit(x_train_PCA,y_train_PCA)
            # sp.fit(x_train_PCA,y_train_PCA)
            # LDA_clf_PCA.fit(x_train_PCA,y_train_PCA)
            Voting1 = VotingClassifier(estimators=[
                ('KNN', KNNModelPCA),  ('LDA', LDA_clf_PCA)],
                voting='soft')
            SVM_clf.fit(x_train,y_train)
            SVM_clfPCA.fit(x_train_PCA,y_train_PCA)
            KNNModel.fit(x_train,y_train)
            KNNModelPCA.fit(x_train_PCA,y_train_PCA)
            Voting1.fit(x_train_PCA,y_train_PCA)
            sp.fit(x_train_PCA,y_train_PCA)

            whetherBreak = 0

            for t in range(len(x_train)):
                totalPre = []
                pre1 = SVM_clf.predict_proba([x_train[t]])
                pre2 = SVM_clfPCA.predict_proba([x_train_PCA[t]])
                pre3 = KNNModel.predict_proba([x_train[t]])
                pre4 = KNNModelPCA.predict_proba([x_train_PCA[t]])
                pre5 = Voting1.predict_proba([x_train_PCA[t]])
                pre6 = sp.predict([x_train_PCA[t]])
                totalPre.append(pre1)
                totalPre.append(pre2)
                totalPre.append(pre3)
                totalPre.append(pre4)
                totalPre.append(pre5)
                totalPre=np.array(totalPre)
                totalPre=totalPre.reshape(60,)
                totalProbability.append(totalPre)
            for u in range(len(x_test)):
                totalPrefortest = []
                pre1 = SVM_clf.predict_proba([x_test[u]])
                pre2 = SVM_clfPCA.predict_proba([x_test_PCA[u]])
                pre3 = KNNModel.predict_proba([x_test[u]])
                pre4 = KNNModelPCA.predict_proba([x_test_PCA[u]])
                pre5 = Voting1.predict_proba([x_test_PCA[u]])
                pre6 = sp.predict([x_test_PCA[u]])
                totalPrefortest.append(pre1)
                totalPrefortest.append(pre2)
                totalPrefortest.append(pre3)
                totalPrefortest.append(pre4)
                totalPrefortest.append(pre5)
                totalPrefortest=np.array(totalPrefortest)
                totalPrefortest=totalPrefortest.reshape(60,)
                print(totalPrefortest.shape)
                totalProbabilityForTest.append(totalPrefortest)


            polymerID = to_categorical(polymerID)
            x_train2, x_test2, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.3,
                                                                  random_state=i)

            x_train =numpy.array( totalProbability)

            x_test = numpy.array( totalProbabilityForTest)



            def deep_model(feature_dim, label_dim):
                from keras.models import Sequential
                from keras.layers import Dense
                model = Sequential()
                print("create model. feature_dim ={}, label_dim ={}".format(feature_dim, label_dim))
                model.add(Dense(64, activation='relu', input_dim=feature_dim))
                model.add(Dense(64, activation='relu'))
                model.add(Dense(64, activation='relu'))
                model.add(Dense(label_dim, activation='softmax'))
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                return model


            def train_deep(X_train, y_train, X_test, y_test):
                feature_dim = X_train.shape[1]
                label_dim = 12

                model = deep_model(feature_dim, label_dim)
                model.summary()
                model.fit(X_train, y_train, batch_size=32, epochs=200, validation_data=(X_test, y_test))
                return model


            model = train_deep(x_train, y_train, x_test, y_test)
            results=model.predict(x_test)
            results=results.tolist()
            for item in results:
                preID.append(item.index(max(item)))
    #         for j in range(len(y_test)):
    #             y_test[j]=int(y_test[j])
    #
            cm = confusion_matrix(y_test_PCA, preID)
    #
            cmtemp = np.zeros((12, 12))
            if len(cm) == 11:
                print('stop')
                continue
                # for i in range(len(cm)):
                #     cmtemp[i]=np.append(cm[i],[0],axis=0)
                #     # cmtemp[i]+=cm[i]
                # cmtemp[11,11]=2
                # print('tempCM；',cmtemp)
                # cmTotal+=cmtemp
                # continue
            cmTotal = cmTotal + cm
            scores = utils.printScore(y_test_PCA, preID)
            print(scores)
            m += 1
            print(m)
            scoreTotal += scores
    #         rep = classification_report(y_test, preID, target_names=PN, output_dict=True)
    #         utils.mkdir('workflow4_report')
    #         workflow4_report = pd.DataFrame(rep)
    #         print(workflow4_report)
    #         workflow4_report.to_csv('workflow4_report/workflow' + str(i) + '.csv')
    #
    #         # t = classification_report(int(y_test), preID, target_names=PN, output_dict=True)
    #         #
    #         # SVM_report = pd.DataFrame(t)
    #         # print(SVM_report)
    #
    #
        print(scoreTotal / m)
        maxnumber.append(sum(scoreTotal / m) )
    #
        cmTotal = cmTotal / m
        print(cmTotal / m)
        utils.plot_confusion_matrix(cmTotal,PN,'Ensemble Learning')
    #
    indexVector=maxnumber.index(max(maxnumber))
    print(max(maxnumber),indexVector)
    # print(randomlist[indexVector])
