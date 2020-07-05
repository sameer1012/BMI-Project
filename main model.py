#Importing Libraries

import pandas as pd
import pickle

data =  pd.read_csv('C:\\Users\\ASUS\\Desktop\\HEROKU\\bmi.csv')

#Checking first few rows in dataset
print(data.head())
print("Rows,Columns :" )
print(data.shape)

#Checking NULL Values if any in dataset
print("Null values : ", data.isnull().any())

#Converting Gender to numerical MALE=1 AND FEMALE=0
def convert_to_int(word):
    word_dict = {'Male':1, 'Female':0, 0: 0}
    return word_dict[word]
data['Gender'] = data['Gender'].apply(lambda x:convert_to_int(x))


print(data.head())

#Adding one extra column to define status of person
def convert_to_status(x):
    index = {0:0,1:1,2:2,3:3,4:4,5:5,0:0}
    return index[x]
data['Status'] = data['Index'].apply(lambda x:convert_to_status(x))

print(data.head(20))


X = data.iloc[:, :3]
print(X.head())

y=data.iloc[:,-1]
print(y.head())


#Applying Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)


# Saving model to disk
pickle.dump(regressor, open('C:\\Users\\ASUS\\Desktop\\model1.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('C:\\Users\\ASUS\\Desktop\\HEROKU\\Upload\\model1.pkl','rb'))
print(model.predict([[1, 172, 139]]).round())

