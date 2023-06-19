import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import  recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

class FTIRByRF:
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
        PN = []
        for item in polymerName:
            if item not in PN:
                PN.append(item)
        pList = list(set(polymerName))
        for item in pList:
            #print(item)
            polymerNameList.append(item)
        #print(len(pList))
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
        polymerID=dataset.iloc[1:,1764]
        polymerID=np.array(polymerID)
        intensity = np.array(intensity)
        return polymerName, waveLength, intensity, polymerID,PN
    def spectrumRdF(x,y):

        clf = RandomForestClassifier(n_estimators= 190,oob_score=True,
                                     random_state=10,max_depth=11,
                                    min_samples_split=2,min_samples_leaf=2,max_features=9)
        clf.fit(x,y)
        # param_test1 = {'max_features':range(3,14,2)}
        # gsearch1 = GridSearchCV(estimator=RandomForestClassifier(n_estimators= 190,oob_score=True, random_state=10,max_depth=11,
        #                                                          min_samples_split=2,min_samples_leaf=2  ),
        #                         param_grid=param_test1,iid=False, scoring='accuracy', cv=5)
        # gsearch1.fit(x, y)
        # print(gsearch1.best_params_, gsearch1.best_score_)
        return clf

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

    def getAccuracy(x_train,y_train,x_test,y_test,PN):

        accuracies = []
        #index = []
        model =FTIRByRF.spectrumRdF(x_train, y_train)
        #index.append(i)
        y_predict = model.predict(x_test)
        t = classification_report(y_test, y_predict, target_names=PN, output_dict=True)
        RandomForest_report = pd.DataFrame(t)
        RandomForest_report.to_csv('RandomForest_report5.csv')
        #print(y_predict)
        sum = len(y_predict)
        correct = 0
        for j in range(len(y_predict)):
            if y_predict[j] == y_test[j]:
                correct = correct + 1
        accuracies.append(correct / sum)
        reScore = recall_score(y_test, y_predict, labels=None, pos_label=1, average=None,
                               sample_weight=None)
        # print(index)
        print('RF score'+str(model.score(x_test,y_test)))
        cm=confusion_matrix(y_test,y_predict)


        #recall score
        for i in range(len(reScore)):
            print(str(i)+':'+str(reScore[i]))
        return correct / sum,cm
if __name__ == '__main__':
    tR=FTIRByRF
    polymerName, waveLength, intensity, polymerID,PN= tR.parseData2('D4_4_publication5.csv', 2, 1763)
    print(intensity.shape)
    print(polymerID.shape)

    x_train, x_test, y_train, y_test = train_test_split(intensity, polymerName, test_size=0.3, random_state=1)
    #clf.fit(intensity,polymerID)
    index=[]
    acc=[]



    # for i in range(30):
    #     index.append(i+1)
    #     acc.append(tR.getAccuracy(x_train,y_train,x_test,y_test,PN))
    # print(max(acc),min(acc),sum(acc)/max(index))
    acc,cm=tR.getAccuracy(x_train,y_train,x_test,y_test,PN)
    tR.plot_confusion_matrix(cm,PN,'Random Forest')
    plt.show()
    # fig, ax = plt.subplots()
    # ax.set_xlabel('i')
    # ax.set_ylabel('accuracy')
    # ax.plot(index, acc, 'b', label='RandomForest', linestyle='dashdot')
    # ax.legend()
    # plt.show()