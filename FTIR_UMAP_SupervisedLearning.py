import umap
import  pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis
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
    plt.xticks(num_local, labels_name, rotation=90,size=15)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name,size=15)  # 将标签印在y轴坐标上
    plt.ylabel('True label',font2)
    plt.xlabel('Predicted label',font2)
    plt.show()
#color = ['r','g','c','y','m','b','pink','maroon','tomato','peru','lawngreen','gold','aqua','dodgerblue']
def getPN(fileName):
    dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False, low_memory=False)

    polymerName = dataset.iloc[1:, 1]
    PN = []
    for item in polymerName:
        if item not in PN:
            PN.append(item)
    return PN
def parseData2(fileName, begin, end):
    dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False)
    polymerName = dataset.iloc[1:971, 1]
    waveLength = dataset.iloc[0, begin:end]
    intensity = dataset.iloc[1:971, begin:end]
    polymerName = np.array(polymerName)
    waveLength = np.array(waveLength)
    polymerNameID = []
    PN=[]
    for item in polymerName:
        if item not in PN:
            PN.append(item)
    PN=np.array(PN)
    for i in range(len(PN)):
        polymerNameID.append(i + 1)
    polymerNameData = []
    for item in polymerName:
        for i in range(len(PN)):
            if item == PN[i]:
                polymerNameData.append(i + 1)

    intensity = np.array(intensity)
    return polymerName, waveLength, intensity, polymerNameData
#reducer = umap.UMAP(random_state=42,n_components=260)
polymerName, waveLength, intensity, polymerNameData= parseData2('D4_4_publication5.csv', 2, 1763)
UMAPList=[]
for i in range(1,400):
    UMAPscore=[]
    reducer1 = umap.UMAP(random_state=42,n_components=(i+1))

    embedding = reducer1.fit_transform(intensity)

    colour= ['c', 'b', 'g', 'm', 'y', 'k', 'darkviolet','olive',
             'tan','lightgreen','gold','cyan','magenta','pink',
             'crimson','navy','cadetblue']
    colour=np.array(colour)
    cols=[]
    # fig,ax = plt.subplots()
    # PN = getPN('D4_4_publication5.csv')
    # tps=[]
    # for i in range(len(PN)):
    #     tp=[]
    #     #sc=''
    #     for j in range(len(embedding)):
    #         if int(polymerNameData[j])==i+1:
    #             sc=colour[i]
    #             tp.append((embedding[j,0],embedding[j,1]))
    #             #print(axis_x1[j])
    #             #print(tp)
    #
    #     tp=np.array(tp)
    #     tps.append(tp)
    # tps=np.array(tps)
    #
    #
    # cols=[]
    # types=[]
    # for i in range(len(PN)):
    #     cols.append(colour[i])
    # for i in range(len(tps)):
    #     types.append(ax.scatter(tps[i][:,0],tps[i][:,1], c=cols[i], marker='o',s=10))
    # plt.title('UMAP')
    # plt.legend(types,PN)
    # plt.show()
    modelMLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(128,128), random_state=1)
    x_train, x_test, y_train, y_test = train_test_split(embedding, polymerNameData, test_size=0.3, random_state=1)
    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    modelMLP.fit(x_train, y_train)
    modelLDA = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                               solver='svd', store_covariance=False, tol=0.001)
    modelSVM = svm.SVC(C=1, kernel='linear', decision_function_shape='ovo')
    modelKNN = KNeighborsClassifier(n_neighbors=5)
    # modelRF= RandomForestClassifier(n_jobs=-1)
    # modelRF.fit(x_train,y_train)
    modelLDA.fit(x_train,y_train)
    modelKNN.fit(x_train, y_train)
    modelSVM.fit(x_train, y_train)
    print('UMAP-MLP:'+str(i*10+10)+':'+str(modelMLP.score(x_test,y_test)))
    print('UMAP-SVM:'+str(i*10+10)+':'+str(modelSVM.score(x_test,y_test)))
    print('UMAP-KNN:'+str(i*10+10)+':' + str(modelKNN.score(x_test, y_test)))
    print('UMAP-LDA:'+str(i*10+10)+':'+str(modelLDA.score(x_test,y_test)))
    UMAPscore.append(modelSVM.score(x_test, y_test))
    UMAPscore.append(modelKNN.score(x_test, y_test))
    UMAPscore.append(modelLDA.score(x_test, y_test))
    UMAPList.append(UMAPscore)
UMP_supervised_report = pd.DataFrame(UMAPList)
UMP_supervised_report.to_csv('UMAP_supervise_report.csv')
    #print('UMAP-RF:' +str(i*10+10)+':'+ str(modelRF.score(x_test, y_test)))



