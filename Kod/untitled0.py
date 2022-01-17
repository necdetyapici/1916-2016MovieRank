
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
 
mainDf = pd.read_csv("movie_metadata.csv",
                     usecols =['movie_title','director_name','color','duration','actor_1_name','language','country','title_year'])  



print(mainDf.shape)  #Satır Sutun sayısı
    
print(mainDf.info) # Tablo bilgileri

print(mainDf.isnull().sum()) # NaN değerler kontrol edildi.

mainDf = mainDf.dropna(axis=0)  #NaN değerler silindi.

print(mainDf.shape) #Son Satır Sutun Durum

print(mainDf.dtypes) #Veri Türlerine bakıldı.



fig1 , ax1 = plt.subplots(1, 1, figsize= (10,5))                     #Yönetmenlere göre film sayısı                  
mainDf['director_name'].value_counts().plot(kind='bar', ax=ax1)     
ax1.set_title('Yönetmenlerin film sayısı',
              fontsize=16)
plt.legend()
fig1.tight_layout()



fig2 , ax2 = plt.subplots(1, 1, figsize= (10,5))                      #Dillere göre film sayısı               
mainDf['language'].value_counts().plot(kind='bar', ax=ax2)     
ax2.set_title('Dillere göre film sayısı',
              fontsize=16)
plt.legend()
fig2.tight_layout()




fig3 , ax3 = plt.subplots(1, 1, figsize= (10,5))                    #Yıllara göre film sayısı                  
mainDf['title_year'].value_counts().plot(kind='bar', ax=ax3)     
ax3.set_title('Yıllara göre film sayısı',
              fontsize=16)
plt.legend()
fig3.tight_layout()


le = preprocessing.LabelEncoder()                           #Verileri Sayısallaştırıldı.         
dtype_object=mainDf.select_dtypes(include=['object'])
for x in dtype_object.columns:
    mainDf[x]=le.fit_transform(mainDf[x])

print(mainDf.dtypes)            #Son duruma göre veri tipleri



X = mainDf.iloc[:,:8].values
y = mainDf['country'].values           
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) #Eğitim ve test için verileri %75, %25 böldük.


sc = StandardScaler()                               #Sayısal verileri ölçeklendirdik.
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

algorithms=[]                                       #Algoritmaların başarı durumlarını grafikleştirmek için 2 değişken tanımladık.
score=[]  



#KNN

knn=KNeighborsClassifier(n_neighbors=9, algorithm='kd_tree') 
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)                                
ac = accuracy_score(y_test, y_pred)                                     
cr = classification_report(y_test, y_pred, labels=np.unique(y_pred)) 
algorithms.append("KNN")
score.append(ac)


#Navie-Bayes

nb=GaussianNB()
nb.fit(X_train,y_train)
y_pred=nb.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
cr = classification_report(y_test, y_pred, labels=np.unique(y_pred))
algorithms.append("Navie-Bayes")
score.append(ac)


##Support Vector Machine

svm = SVC(random_state=1,kernel='linear', C=1, degree=3, gamma='scale', max_iter=-1)
svm.fit(X_train,y_train)
y_pred = svm.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
cr = classification_report(y_test, y_pred, labels=np.unique(y_pred))
algorithms.append("Support Vector Machine")
score.append(ac)


#DecisionTree

dt=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=None)
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
cr = classification_report(y_test, y_pred, labels=np.unique(y_pred))
algorithms.append("Decision Tree")
score.append(ac)


# LogisticRegression


lr = LogisticRegression(solver='lbfgs')
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
cr = classification_report(y_test, y_pred, labels=np.unique(y_pred))
algorithms.append("Logistic Regression")
score.append(ac)


# Yapay Sinir Ağları


sknet = MLPClassifier(hidden_layer_sizes=(8), learning_rate_init=0.01, max_iter=100)
sknet.fit(X_train, y_train)
y_pred = sknet.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
cr = classification_report(y_test, y_pred, labels=np.unique(y_pred))
algorithms.append("Yapay Sinir Ağları")
score.append(ac)



