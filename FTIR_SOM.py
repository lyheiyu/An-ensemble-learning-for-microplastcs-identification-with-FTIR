from minisom import  MiniSom
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import  numpy as np
from sklearn import svm
def parseData2(fileName, begin, end):
    dataset = pd.read_csv(fileName, header=None, encoding='latin-1', keep_default_na=False, low_memory=False)

    polymerName = dataset.iloc[1:, 1]
    waveLength = dataset.iloc[0, begin:end]
    intensity = dataset.iloc[1:, begin:end]
    polymerName = np.array(polymerName)
    waveLength = np.array(waveLength)

    polymerID = dataset.iloc[1:, 1764]
    polymerID = np.array(polymerID,dtype=np.float32)
    print(polymerID)
    print(max(polymerID))
    polymerNameID = []
    PN = []
    for item in polymerName:
        if item not in PN:
            PN.append(item)
    PN = np.array(PN)
    for i in range(len(PN)):
        polymerNameID.append(i + 1)
    polymerNameData = []
    for item in polymerName:
        for i in range(len(PN)):
            if item == PN[i]:
                polymerNameData.append(i + 1)
    intensity = np.array(intensity, dtype=np.float64)
    return polymerName, waveLength, intensity, polymerNameData


polymerName, waveLength, intensity, polymerID = parseData2('D4_4_publication5.csv', 2, 1763)
pList = list(set(polymerName))

x_train, x_test, y_train, y_test = train_test_split(intensity, polymerName, test_size=0.3, random_state=1)
print(x_train.shape[1])
somList=[]
for j in range(50):
    somScore=[]
    som=MiniSom(x=j+1,y=j+1, input_len=intensity.shape[1],sigma=0.1,learning_rate=0.2)
    som.random_weights_init(intensity)
    starting_weights= som.get_weights().copy()
    som.train_random(intensity,1000)
    qnt=som.quantization(intensity)
    clustered=np.zeros(intensity.shape)
    for i, q in enumerate(qnt):
      clustered[np.unravel_index(i, dims=(intensity.shape[0]))] = q
    x_train, x_test, y_train, y_test = train_test_split(clustered, polymerID, test_size=0.3, random_state=1)

    print(clustered.shape)
    def spectrumSVM(x, y, cvalue, kModel, decFunction):
        model = svm.SVC(C=cvalue, kernel=kModel, decision_function_shape=decFunction)

        model.fit(x, y)

        return model
    def spectrumLDA(x,y):
        model=LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                  solver='svd', store_covariance=False, tol=0.001)

        model.fit(x,y)

        return model
    KNNModel= KNeighborsClassifier(n_neighbors=5)
    KNNModel.fit(x_train,y_train)
    SVMmodel=spectrumSVM(x_train,y_train,1,'linear','ovo')
    LDAModel=spectrumLDA(x_train,y_train)
    somScore.append(SVMmodel.score(x_test, y_test))
    somScore.append(KNNModel.score(x_test, y_test))
    somScore.append(LDAModel.score(x_test, y_test))
    somList.append(somScore)
    print(j)
    print(SVMmodel.score(x_test,y_test))
    print(KNNModel.score(x_test,y_test))
    print(LDAModel.score(x_test,y_test))
SOM_supervise_report=pd.DataFrame(somList)
SOM_supervise_report.to_csv('SOM_supervise_report.csv')