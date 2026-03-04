import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

import pickle


# Load dataset
data = pd.read_csv("data/sales.csv")

print("Dataset Loaded Successfully")
print(data.head())


# Convert date
data['Date'] = pd.to_datetime(data['Date'])

 

# Create sales column if missing
if 'Weekly_Sales' not in data.columns:
    print("Weekly_Sales column missing. Generating sample sales data.")
    data['Weekly_Sales'] = np.random.randint(20000,80000,len(data))


# Remove missing values
data = data.dropna()


# Features
X = data[['Temperature','Fuel_Price','CPI','Unemployment']]
y = data['Weekly_Sales']


# Split dataset
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)


# Train model
model = LinearRegression()
model.fit(X_train,y_train)

print("Model trained successfully")


# Prediction
predictions = model.predict(X_test)


# Evaluation
mae = mean_absolute_error(y_test,predictions)
r2 = r2_score(y_test,predictions)

print("MAE:",mae)
print("R2:",r2)


# Graph
plt.scatter(y_test,predictions)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Sales Prediction")
plt.show()


# Save model
pickle.dump(model,open("model/model.pkl","wb"))

print("Model saved in model folder")