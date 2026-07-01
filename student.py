import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    "Study_Hours":[2,4,6,8,10],
    "Attendance":[60,70,80,90,95],
    "Marks":[40,55,70,85,95]
}

df = pd.DataFrame(data)

X = df[["Study_Hours","Attendance"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X,y)

prediction = model.predict([[7,85]])
print("Predicted Marks:", prediction)