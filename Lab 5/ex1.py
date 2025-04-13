"""
Classifying flowers into 3 classes, with a real data set

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# df stands for data-frame
df = pd.read_csv("iris/iris.data", header=None)
X = df.iloc[:, :4]  # take all rows and columns from 0 to 4 (exclusive of 4)

classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'] # we could have not gone with hardcoded values but we're lazy lmao
T = df[4].replace(classes,[0,1,2]) #define the target values for each class

plt.close('all')
df.columns=['SepalLength', 'SepalWidth', 'PetalLength', 'Iris-Virginica', 'Class']

# Create pairplot to visualize relationships between features
sns.pairplot(df, hue='Class')
plt.show()

"""
Split the data into training and testing sets

For small data sets, if you have some bad points (bad examples) they will affect quite a lot the
training process if they are inserted into the training set. Play with random_state to split data
so that your bad points from the data set are inserted into the testing set 
"""
X_train, X_test, y_train, y_test = train_test_split(X, T, test_size=0.2, random_state=2)


# Create and train the MLP classifier
"""
Alpha is learning rate
verbose is for detailed output
we have 3 layers, 3 neurons on first hidden layer, 2 layers on 2nd hidden layer, 3 neurons on output layer
"""
mlp = MLPClassifier(hidden_layer_sizes=(3, 2),solver='sgd',alpha=1e-5,verbose=1, max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

"""
3 biases for first layer, 2 for 2nd layer, 3 again for 3rd layer
3x4 weights for first layer (3 neurons and 4 inputs results into 12 total weights) and so on for the rest
3x2 2nd layer
2x3 3rd layer
"""

biases=mlp.intercepts_
weights=mlp.coefs_

# Make predictions
y_pred = mlp.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Print detailed classification report
print("\nClassification Report:")
print('The accuracy is: ', accuracy) # accuracy score
print('Confusion Matrix is: ')
print(confusion_matrix(y_train, y_test))

"""

"""
sns.pairplot ( df, hue='Class')


"""
Split the the input data into training data and testing, around 70% of input for training, 30% for testing
but do it in a way that the MLP will receive inputs randomly, so that we dont take the input data for
the first 2 classes only
"""

