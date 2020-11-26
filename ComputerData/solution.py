import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import joblib 	


data=pd.read_csv('Computer_Data.csv')
print(data.corr())
y=data.price
x=data.drop(['price'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=False)
# regr=linear_model.LinearRegression()
# regr.fit(x_train,y_train)
# joblib.dump(regr,'model_joblib')
model=joblib.load('model_joblib')
predi=model.predict(x_test)
print('Coefficient of determination: %.2f'%r2_score(y_test, predi))