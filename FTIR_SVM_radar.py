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
    return np.mean(dataX,axis=0)#axis=0表示按照列来求均值，如果输入list,则axis=1
def pca(XMat, k):
    average = meanX(XMat)

    m,n = np.shape(XMat)
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T)   #计算协方差矩阵
    featValue, featVec=  np.linalg.eig(covX)  #求解协方差矩阵的特征值和特征向量
    index = np.argsort(-featValue) #按照featValue进行从大到小排序
    finalData = []
    if k > n:
        print ("k must lower than feature number")
        return
    else:
        #注意特征向量时列向量
        selectVec = np.matrix(featVec.T[index[:k]]) #所以这里需要进行转置，而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
        print( '############',selectVec[0].reshape(1,-1))
        PC1=selectVec[0].reshape(1761)
        print(PC1.shape)
        finalData = data_adjust * selectVec.T
        reconData = (finalData * selectVec) + average
    return finalData, reconData,selectVec
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
    x_class=[]
    y_class=[]
    for i in range(12):
        m=[]
        z=[]
        for j in range(len(intensity)):
            if int(polymerID[j])==i+1:
                z.append(polymerID[j])
                m.append(intensity[j])
        x_class.append(m)
        y_class.append(z)

    return polymerName, waveLength, intensity, polymerID,x_class,y_class

polymerName, waveLength, intensity, polymerID = TestKNN.parseData2('D4_4_publication5.csv', 2, 1763)
def plotBestFit(data1, data2):
    dataArr1 = np.array(data1)
    #dataArr2 = np.array(data2)

    m = np.shape(dataArr1)[0]
    axis_x1 = []
    axis_y1 = []
    axis_x2 = []
    axis_y2 = []
    for i in range(m):
        axis_x1.append(dataArr1[i,0])
        axis_y1.append(dataArr1[i,1])
        #axis_x2.append(dataArr2[i,0])
        #axis_y2.append(dataArr2[i,1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(axis_x1, axis_y1, s=50, c='red', marker='s')
   # ax.scatter(axis_x2, axis_y2, s=50, c='blue')
    plt.xlabel('x1'); plt.ylabel('x2');
    #plt.savefig("outfile.png")
    plt.show()
def main(i):
    datafile = "data.txt"
    polymerName, waveLength, intensity, polymerID,x_class,y_class = parseData2('D4_4_publication5.csv', 2, 1763)

    k = (i+1)
    return pca(intensity, k)
if __name__ == '__main__':
    finalData, reconMat, Vector = main(129)

    finalData = np.array(finalData, dtype=np.float32)
    print(finalData.shape)
    # ohe = sp.OneHotEncoder(sparse=False, dtype="int32")
    # result = ohe.fit_transform(finalData)
    # finalData = MMScaler.fit_transform(finalData)
    # modelMLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128,128), random_state=1)
    polymerName, waveLength, intensity, polymerID,x_class,y_class = parseData2('D4_4_publication5.csv', 2, 1763)
    x_train, x_test, y_train, y_test = train_test_split(finalData, polymerID, test_size=0.3, random_state=1)
    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    #x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.3, random_state=1)
    knn_clf_PCA = KNeighborsClassifier(n_neighbors=5)
    RandomF_PCA = RandomForestClassifier(n_jobs=-1)
    SVM_clf_PCA=svm.SVC(C=0.3, kernel='linear', decision_function_shape='ovo')
    LDA_clf_PCA=LinearDiscriminantAnalysis(n_components=10, priors=None, shrinkage=None,
                  solver='svd', store_covariance=True, tol=0.0001)
    #voting_clf=VotingClassifier(estimators=[('knn',knn_clf),('Randomforest',RandomF),('SVM',SVM_clf),('LDA',LDA_clf)])
            #SVM_clf = svm.SVC(probability=True)

        #PLS = PLSRegression(n_components=10)
            #voting_clf = VotingClassifier(estimators=[('knn', knn_clf), ('Randomforest', RandomF), ('SVM', SVM_clf)],
               #                           voting='soft')
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    RandomF = RandomForestClassifier(n_jobs=-1)
    SVM_clf=svm.SVC(C=0.3, kernel='linear', decision_function_shape='ovo')
    LDA_clf=LinearDiscriminantAnalysis(n_components=10, priors=None, shrinkage=None,
                  solver='svd', store_covariance=True, tol=0.0001)
    classfiers = [SVM_clf,]
    classfiersPCA=[SVM_clf_PCA,knn_clf_PCA,RandomF_PCA,LDA_clf_PCA]
    labels = []
    scorelist = []
    for reg in classfiers:
        thisscore = []
        nowT=time()
        print(nowT)
        reg.fit(x_train, y_train)
        y_pred = reg.predict(x_test)
        labels.append(reg.__class__.__name__+'PCA')
        for score in scores:
            thisscore.append(score(y_test, y_pred))
            print(reg.__class__.__name__+'---'+score.__name__+':',score(y_test, y_pred))
        #print(reg.__class__.__name__,score.__name__, score(y_test, y_pred))
        print(reg.__class__.__name__+'Time:',time()-nowT)
        # y_score = reg.predict_proba(x_test)
        # y_score = np.array(y_score)
        y_test = np.array(y_test)
        scorelist.append(thisscore)
    # sp = PLSDA(n_components=50, style='hard')
    # y = []
    # for item in y_train:
    #     y.append(np.int32(item))
    # y2 = []
    # for item in y_test:
    #     y2.append(np.int32(item))
    # y = np.array(y)
    # y2 = np.array(y2)
    # PLSDATime=time()
    # _ = sp.fit(x_train, y)
    # # _ = sc.fit(x_trai y_train)
    # pred = sp.predict(x_test)
    # p = np.array(pred)
    # p = p.reshape(p.shape[0])
    # thisscore = []
    # for score in scores:
    #     thisscore.append(score(y2,pred))
    # scorelist.append(thisscore)
    # print('PLSDA:',time()-PLSDATime)
    # labels.append('PLSDA')
    # #df, I, CSNS, CSPS, CEFF, TSNS, TSPS, TEFF = sp.figures_of_merit(pred, y2)
    # score = sp.score(x_test, y2)
    # print('scorePLSDA:' + str(score))
    polymerName, waveLength, intensity, polymerID,x_each,y_each = parseData2('D4_4_publication5.csv', 2, 1763)
    #x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.3, random_state=1)
    # matrixC=[]
    # y_true=[]
    # ypre=[]
    # PN=[]
    # for item in polymerName:
    #     if item not in PN:
    #         PN.append(item)
    # PN=np.array(PN)
    # for i in range(len(PN)):
    #     #x_train, x_test, y_train, y_test = train_test_split(x_each[i], y_each[i], test_size=0.3, random_state=1)
    #     sc = SIMCA(n_components=5, alpha=0.05)
    #     _ = sc.fit(x_each[i], y_each[i])
    #     xe = []
    #
    #     pred = sc.predict(x_each[i])
    #
    #     x=0
    #     for item in pred:
    #         y_true.append(True)
    #         ypre.append(item)
    #         if str(item)=='True':
    #             x+=1
    #     print(str(i+1)+'class:'+'accuracy：'+str(x/np.array(pred).shape[0]))
    # thisscore = []
    # for score in scores:
    #     thisscore.append(score(y_true,ypre))
    # scorelist.append(thisscore)
    # labels.append('SIMCA')
    #data_labels = labels
    x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.3, random_state=1)
    # colors = ['c', 'b', 'g', 'r', 'm', 'y', '#377eb8', 'darkviolet', 'olive',
    #               'tan', 'lightgreen', 'gold', 'cyan', 'magenta', 'pink',
    #               'crimson', 'navy', 'cadetblue']


    # for reg in classfiers:
    #     thisscore = []
    #     nowT=time()
    #     print(nowT)
    #     reg.fit(x_train, y_train)
    #     y_pred = reg.predict(x_test)
    #     labels.append(reg.__class__.__name__)
    #     for score in scores:
    #         thisscore.append(score(y_test, y_pred))
    #         print(reg.__class__.__name__+'---'+score.__name__+'PCA'+':',score(y_test, y_pred))
    #     #print(reg.__class__.__name__,score.__name__, score(y_test, y_pred))
    #     print(reg.__class__.__name__+'PCA'+'Time:',time()-nowT)
    #     # y_score = reg.predict_proba(x_test)
    #     # y_score = np.array(y_score)
    #     y_test = np.array(y_test)
    #     scorelist.append(thisscore)
    # sp = PLSDA(n_components=50, style='hard')
    # y = []
    # for item in y_train:
    #     y.append(np.int32(item))
    # y2 = []
    # for item in y_test:
    #     y2.append(np.int32(item))
    # y = np.array(y)
    # y2 = np.array(y2)
    # PLSDATime=time()
    # _ = sp.fit(x_train, y)
    # # _ = sc.fit(x_trai y_train)
    # pred = sp.predict(x_test)
    # p = np.array(pred)
    # p = p.reshape(p.shape[0])
    # thisscore = []
    # for score in scores:
    #     thisscore.append(score(y2,pred))
    # scorelist.append(thisscore)
    # # print('PLSDA:',time()-PLSDATime)
    # labels.append('PLSDA')
    # #df, I, CSNS, CSPS, CEFF, TSNS, TSPS, TEFF = sp.figures_of_merit(pred, y2)
    # score = sp.score(x_test, y2)
    # print('scorePLSDA:' + str(score))
    # polymerName, waveLength, intensity, polymerID,x_each,y_each = parseData2('D4_4_publication5.csv', 2, 1763)
    # #x_train, x_test, y_train, y_test = train_test_split(intensity, polymerID, test_size=0.3, random_state=1)
    # matrixC=[]
    # y_true=[]
    # ypre=[]
    # PN=[]
    # for item in polymerName:
    #     if item not in PN:
    #         PN.append(item)
    # PN=np.array(PN)
    # for i in range(len(PN)):
    #     #x_train, x_test, y_train, y_test = train_test_split(x_each[i], y_each[i], test_size=0.3, random_state=1)
    #     sc = SIMCA(n_components=5, alpha=0.05)
    #     _ = sc.fit(x_each[i], y_each[i])
    #     xe = []
    #
    #     pred = sc.predict(x_each[i])
    #
    #     x=0
    #     for item in pred:
    #         y_true.append(True)
    #         ypre.append(item)
    #         if str(item)=='True':
    #             x+=1
    #     print(str(i+1)+'class:'+'accuracy：'+str(x/np.array(pred).shape[0]))
    # thisscore = []
    # for score in scores:
    #     thisscore.append(score(y_true,ypre))
    # scorelist.append(thisscore)
    # labels.append('SIMCA')
    data_labels = labels
    colors = [ 'b','r', 'darkviolet', 'g','k' , 'gold', 'cadetblue','magenta', 'pink',
                  'crimson', 'navy', 'cadetblue','darkviolet', 'b', 'g', 'r', 'm', 'y', '#377eb8', 'darkviolet', 'olive',
                'tan', 'lightgreen', 'gold', 'cyan', 'magenta', 'pink',
                'crimson', 'navy', 'cadetblue']
    data = np.array(scorelist)
    radar_labels = np.array([ 'kappa_score','f1', 'accuracy', 'recall', 'precision'])
    scoList=pd.DataFrame(scorelist)
    scoList.to_csv('scoreList.csv')
    plt.rcParams['font.family'] = "SimHei"
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    fig = plt.figure(facecolor="White")
    plt.subplot(111, polar=True)
    # print(data)
    # data=data.T
    # print(data)
    # print(data.shape)
    plt.plot(angles, [1,1,1,1,1], 'o', linewidth=1, alpha=0, color=colors[1])
    for i,item in enumerate(data):
        plt.plot(angles,item,'o', linewidth=1, alpha=1,color=colors[3])
        print(i)
        # if i==1:
        #     plt.fill(angles, item, alpha=0.6, color=colors[i])
        if i==5:
            plt.fill(angles, item, alpha=0.5, color=colors[i])
        # else:
        plt.fill(angles,item, alpha=0.2,color=colors[3])

    # plt.plot(angles, data.T, 'o-', linewidth=1, alpha=0.2,color='navy')
    # plt.fill(angles, data.T, alpha=0.25)
    plt.thetagrids(angles * 180 / np.pi, radar_labels, size=15)
    plt.figtext(0.7, 0.7, 'metrics of PCA-LDA based identification methods', ha='center', size=20)
    legend = plt.legend(data_labels, loc=(0.94, 0.80), labelspacing=0.2)
    plt.setp(legend.get_texts(), fontsize='large')
    plt.grid(True)
    plt.savefig('ML metrics_radar.jpg')
    plt.show()
