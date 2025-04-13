import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier # neural network
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df=pd.read_csv('iris/iris.data',header=None)
X=df.iloc[:,0:4].values

classes=['Iris-setosa','Iris-versicolor','Iris-virginica']
T=df[4].replace(classes,[0,1,2])

plt.close('all')
df.columns=['SepalLength','SepalWidth','PetalLength',
            'PetalWidth', 'Class']
sns.pairplot( df,hue='Class')

xTrain, xTest, tTrain, tTest = train_test_split(X,T, test_size = 0.2,
                                                random_state=1)

net = MLPClassifier(solver='sgd', alpha=1e-5, verbose=1,max_iter=1000, 
                    hidden_layer_sizes=(3, 3), random_state=1)

net.fit(xTrain, tTrain)
# b=net.intercepts_
# w=net.coefs_
yTest=net.predict(xTest)

print('The accuracy is:',accuracy_score(tTest,yTest)) # accuracy_score(y_true, y_pred)
print('Confusion Matrix is: ') 
print(confusion_matrix(tTest,yTest)) # confusion_matrix(y_true, y_pred) - ON LINES!!!