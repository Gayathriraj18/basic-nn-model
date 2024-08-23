# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural Network regression model is a type of machine learning algorithm inspired by the structure of the brain. It excels at identifying complex patterns within data and using those patterns to predict continuous numerical values.his includes cleaning, normalizing, and splitting your data into training and testing sets. The training set is used to teach the model, and the testing set evaluates its accuracy. This means choosing the number of layers, the number of neurons within each layer, and the type of activation functions to use.The model is fed the training data.Once trained, you use the testing set to see how well the model generalizes to new, unseen data. This often involves metrics like Mean Squared Error (MSE) or Root Mean Squared Error (RMSE).Based on the evaluation, you might fine-tune the model's architecture, change optimization techniques, or gather more data to improve its performance.


## Neural Network Model

![image](https://github.com/user-attachments/assets/646f4314-fe96-4481-aff4-4a3ad89fbda5)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Gayathri A
### Register Number: 212221230028
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default
auth.authenticate_user()
creds,_=default()
gc=gspread.authorize(creds)
worksheet=gc.open('e1').sheet1
data=worksheet.get_all_values()
dataset1=pd.DataFrame(data[1:],columns=data[0])
dataset1=dataset1.astype(float)
dataset1.head()
x=dataset1.values
y=dataset1.values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=33)
Scaler=MinMaxScaler()
Scaler.fit(x_train)
x_train=Scaler.transform(x_train)
ai_brain=Sequential([
    Dense(8,activation='relu'),
    Dense(10,activation='relu'),
    Dense(1)
])
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(x_train,y_train,epochs=20)
loss_df=pd.DataFrame(ai_brain.history.history)
loss_df.plot()
ai_brain.evaluate(x_test,y_test)
X_n1 = [[3,5]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)

```
## Dataset Information

![Screenshot 2024-08-19 143033](https://github.com/user-attachments/assets/f507f574-50ea-465e-8595-e2ac274c0879)

## Output Epoch

![Screenshot 2024-08-19 143115](https://github.com/user-attachments/assets/9460faf2-6de4-4cf3-aab4-099d26ff13f4)

### Training Loss Vs Iteration Plot

![Screenshot 2024-08-19 143153](https://github.com/user-attachments/assets/9ba95536-fcb6-4b0e-bcff-f169eac8b758)


### Test Data Root Mean Squared Error

![Screenshot 2024-08-19 143225](https://github.com/user-attachments/assets/8aa1f24c-f075-46c5-b78a-1915055d168f)

### New Sample Data Prediction

![Screenshot 2024-08-19 143251](https://github.com/user-attachments/assets/374fb933-4721-40f7-8253-694e43cbd7de)

## RESULT

Thus a neural network regression model for the given dataset has been developed.
