import pandas as pd
import sklearn.ensemble
from sklearn import preprocessing
labelEncoder=preprocessing.LabelEncoder()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data=pd.read_csv(r"C:\Users\raoar\OneDrive\Desktop\train.csv")
test=pd.read_csv(r"C:\Users\raoar\OneDrive\Desktop\test.csv")
test_ids=test["PassengerId"]
def clear(data):
    data=data.drop(["Ticket","Cabin","Name","PassengerId"],axis=1)
    cols=["SibSp","Parch","Fare","Age"]
    for i in cols:
        data[i].fillna(data[i].median(),inplace=True)
        data.Embarked.fillna("U",inplace=True)
        return data
data=clear(data)
test=clear(test)
columns=["Sex","Embarked"]
for i in columns:
    data[i]=labelEncoder.fit_transform(data[i])
    test[i]=labelEncoder.transform(test[i])
    print(labelEncoder.classes_)
print(data.head(5))
y=data["Survived"]
X=data.drop("Survived",axis=1)
X_train,X_values,y_train,y_values=train_test_split(X,y,test_size=0.2,random_state=42)
clf=sklearn.ensemble.HistGradientBoostingRegressor(random_state=0,max_iter=1000).fit(X_train,y_train)
predictions=clf.predict(X_values)
submission=clf.predict(test)
df=pd.DataFrame({"PassengerId":test_ids.values,
                 "Survived":submission
                 })
df.to_csv("submission.csv",index=False)