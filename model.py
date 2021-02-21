import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df["target"] = iris.target

x = df.drop(["target"],axis = 1)
y = df["target"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

from sklearn.svm import SVC
model = SVC()
model.fit(x_train,y_train)

result = model.predict([[4.7,3.2,1.3,0.2]])
print(result)

import pickle
pickle.dump(model,open("model.pkl","wb"))