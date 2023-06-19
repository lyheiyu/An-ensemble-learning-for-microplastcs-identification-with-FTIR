import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from plsda import  PLSDA
def __init__(self):
    pass
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
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
def parseData2(fileName, begin, end):
    dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False, low_memory=False)

    polymerName = dataset.iloc[1:971, 1]
    waveLength = dataset.iloc[0, begin:end]
    intensity = dataset.iloc[1:971, begin:end]
    polymerName = np.array(polymerName)
    waveLength = np.array(waveLength)

    polymerID = dataset.iloc[1:, 1764]
    polymerID = np.array(polymerID)

    intensity = np.array(intensity)
    return polymerName, waveLength, intensity, polymerID
def spectrumKNN(x,y,number):
    model= KNeighborsClassifier(n_neighbors=number)

    model.fit(x,y)

    return model

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
             'size': 20,
             }
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90, size=15)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name, size=15)  # 将标签印在y轴坐标上
    plt.ylabel('True label', font2)
    plt.xlabel('Predicted label', font2)
    plt.show()

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
    polymerName, waveLength, intensity, polymerID = parseData2('D4_4_publication5.csv', 2, 1763)

    k = i
    return pca(intensity, k)
if __name__ == '__main__':

    finalData, reconMat, Vector = main(120)

    finalData = np.array(finalData, dtype=np.float32)
    #this is for the 216_2018.csv
    #polymerName, polymerID, waveLength, intensity=tk.parseData2('216_2018_1156_MOESM5_ESM.csv',3,1179)
    #this is for the D4 public_csv
    polymerName, waveLength, intensity,polymerID= parseData2('D4_4_publication5.csv', 2, 1763)
    id=[]
    for i in range(len(polymerID)):
        if polymerID[i] not in id:
            id.append(polymerID[i])
    print(np.array(id).shape)
    #pList = list(set(polymerName))
    PN=getPN('D4_4_publication5.csv')
    pList=PN
    print(max(polymerID))
    #print(waveLength)
    #print(intensity.shape,polymerID.shape,polymerName.shape)

    for item in intensity:

        for i in range(len(item)):
            if item[i]=='':

                item[i]=0

                #print(waveIntensity)
                #print(item)
    MaxValueFordataset=[]
    MaxNum=[]

    x_train, x_test, y_train, y_test = train_test_split(finalData, polymerID, test_size=0.3, random_state=1)
    #print(x_train)
    accuracies=[]
    index=[]
    recallScore=[]
    sp = PLSDA(n_components=30, style='hard')
    y = []
    for item in y_train:
        y.append(np.int32(item))
    y2 = []
    for item in y_test:
        y2.append(np.int32(item))
    y = np.array(y)
    y2 = np.array(y2)
    _ = sp.fit(x_train, y)
    # _ = sc.fit(x_trai y_train)
    pred = sp.predict(x_test)
    p = np.array(pred)
    p = p.reshape(p.shape[0])
    # create confusion matrix
    predForCm = []
    for item in p:
        predForCm.append(int(item))

    PN = getPN('D4_4_publication5.csv')
    from sklearn.metrics import classification_report

    t = classification_report(y2, predForCm, target_names=PN, output_dict=True)
    PLSDA_report = pd.DataFrame(t)
    PLSDA_report.to_csv('PCA-PLSDA_report5.csv')
    cm = confusion_matrix(y2, predForCm)
    plot_confusion_matrix(cm, PN, 'PCA-PLSDA')
    cm = pd.DataFrame(cm)
    cm.to_csv('PLSDA-PCA.csv')
    # pred=sc.predict(x_train)
    df, I, CSNS, CSPS, CEFF, TSNS, TSPS, TEFF = sp.figures_of_merit(pred, y2)
    score = sp.score(x_test, y2)
    print('score:---------------------:' + str(score))
    # print(simcaScore)
    """
            Compute figures of merit for PLS-DA approaches as in [1].
            Parameters
            ----------
            predictions : list(list)
                Array of array values containing the predicted class of points (in
                order). Each row may have multiple entries corresponding to
                multiple class predictions in the soft PLS-DA case.
            actual : array-like
                Array of ground truth classes for the predicted points.  Should
                have only one class per point.
            Returns
            -------
            df : pandas.DataFrame
                Inputs (index) vs. predictions (columns).
            I : pandas.Series
                Number of each class asked to classify.
            CSNS : pandas.Series
                Class sensitivity.
            CSPS : pandas.Series
                Class specificity.
            CEFF : pandas.Series
                Class efficiency.
            TSNS : float64
                Total sensitivity.
            TSPS : float64
                Total specificity.
            TEFF : float64
                Total efficiency.
            """
    print('df:' + str(df), 'I:' + str(I), 'CSNS:' + str(CSNS), 'CSPS' + str(CSPS), 'CEFF' + str(CEFF),
          'TSNS' + str(TSNS), 'TSPS:' + str(TSPS), 'TEFF' + str(TEFF))
    ##This is for testing the maxNum and value for various testrate
    # index=[]
    # for i in range(len(MaxNum)):
    #     index.append(i+1)
    # fig, ax = plt.subplots()
    # ax.set_xlabel('Max K value')
    # ax.set_ylabel('accuracy')
    # ax.plot(index, MaxNum, 'y', label='KNN', linestyle='dashdot')
    # plt.clf()
    # plt.plot(index, MaxValueFordataset, 'r', label='KNN', linestyle='dashdot')
    # plt.show()
    # ax.legend()
    # plt.show()
    ####
    # accuracies2 = []
    # index2 = []
    # for i in range(3,40):
    #     model = tk.spectrumKNN(x_train, y_train, i)
    #     index2.append(i)
    #     y_predict2 = model.predict(x_test)
    #     #print(y_predict)
    #     sum = len(y_predict2)
    #     correct = 0
    #     for j in range(len(y_predict2)):
    #         if y_predict2[j] == y_test[j]:
    #             correct = correct + 1
    #     accuracies2.append(correct/sum)
    #
    #     #print(index)
    #     print('accuracy2:' + str(correct / sum))
    #
    # #print(max(accuracies))
    # #for i in range(len(accuracies)):
    # for i in range(len(accuracies2)):
    #     if accuracies2[i]==max(accuracies):
    #         print(accuracies2[i])
    #         print('maxValue2：'+str(index[i]))
    # #fig, ax = plt.subplots()
    # ax.set_xlabel('K value')
    # ax.set_ylabel('accuracy')
    # ax.plot(index2,accuracies2, 'r', label='KNN', linestyle='dotted')
    # #ax.legend()
    # ax.legend()
    # plt.show()
    #scores = cross_val_score(model, x_train, y_train, cv=20, scoring='accuracy')
    #n_scores.append(scores.mean())
    #predict=model.predict([intensity[3]])
    #print(polymerName)

    #print(scores)
    # print(model.score(x_test, y_test))
    # print(model.predict(x_test))
