import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
print("DATA CLASSIFICATION USING AI")
iris=load_iris()
data=pd.DataFrame(iris.data,columns=iris.feature_names)
data["species"]=iris.target
print("\nDataset Loaded Successfully!")
print("\nFirst 5 Rows:")
print(data.head())
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print("\nTraining Data:",len(x_train))
print("Testing Data:",len(x_test))
model=DecisionTreeClassifier()
print("\nTraining Model....")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy= accuracy_score(y_test, y_pred)
print("MODEL PERFORMANCE")
print(f"accuracy: {accuracy * 100:.2f}%")
plt.figure(figsize=(7,5))
plt.scatter(
    iris.data[:,0],
    iris.data[:,1],
    c=iris.target
)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Flower Classification")
plt.show()
print("\nProject Completed Successfully!")
    
