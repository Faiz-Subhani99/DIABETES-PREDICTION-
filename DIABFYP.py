import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle 
df=pd.read_csv(r"C:\Users\Faiz\Desktop\diabetes.csv")
print(df)
dataset = pd.read_csv(r'C:\Users\Faiz\Desktop\diabetes.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
acc=accuracy_score(y_test, y_pred)

pickle.dump(classifier,open('DIAFYP.pkl','wb'))
model=pickle.load(open('DIAFYP.pkl','rb'))
print(model.predict([[0,1,2,3,4,5,6,7]]))
