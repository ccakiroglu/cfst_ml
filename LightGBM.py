from pandas import read_csv, DataFrame
from numpy import absolute,arange,mean,std,argsort,sqrt
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor 
#from xgboost import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from matplotlib import pyplot 
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score, accuracy_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
pyplot.rc('text',usetex=True)
pyplot.rc('font',size=15)
pyplot.figure(figsize=(6.4,4.26666666667))

scaler = MinMaxScaler()
pyplot.rc('text',usetex=True)
pyplot.rc('font',size=15)
colnames = ['B','H','t','L','fy','fc','Ntest']
path='path to inputfile.csv'
df = read_csv(path,header=0,names=colnames)
Nu0_2500dir='path to \\Nu_0_2500.csv'
range1dir='path to \\fc_0_25.csv'
range2dir='path to \\fc_25_50.csv'
range3dir='path to \\fc_50_100.csv'
range4dir='path to \\fc_100Plus.csv'
df = read_csv(path,header=0,names=colnames)
df0_2500 = read_csv(Nu0_2500dir,header=0,names=colnames)
dfRange1 = read_csv(range1dir,header=0,names=colnames)
dfRange2 = read_csv(range2dir,header=0,names=colnames)
dfRange3 = read_csv(range3dir,header=0,names=colnames)
dfRange4 = read_csv(range4dir,header=0,names=colnames)
data=df.values
data0_2500=df0_2500.values
dataR1=dfRange1.values
dataR2=dfRange2.values
dataR3=dfRange3.values
dataR4=dfRange4.values
# split into inputs and outputs
#X, y = df.iloc[:, :-1], df.iloc[:, -1]
X, y = data[:, :-1], data[:, -1]
#X_7500Plus, y_7500Plus = data7500Plus[:, :-1], data7500Plus[:, -1]
#X_5000_7500, y_5000_7500 = data5000_7500[:, :-1], data5000_7500[:, -1]
#X_2500_5000, y_2500_5000 = data2500_5000[:, :-1], data2500_5000[:, -1]
X_0_2500, y_0_2500 = data0_2500[:, :-1], data0_2500[:, -1]
#X_25_50, y_25_50 = dataR2[:, :-1], dataR2[:, -1]
#X_50_100, y_50_100 = dataR3[:, :-1], dataR3[:, -1]
#X_100Plus, y_100Plus = dataR4[:, :-1], dataR4[:, -1]
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_0_2500 = scaler.transform(X_0_2500)
#X_2500_5000 = scaler.transform(X_2500_5000)
#X_5000_7500 = scaler.transform(X_5000_7500)
#X_7500Plus = scaler.transform(X_7500Plus)
LGBMmodel = LGBMRegressor()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++
cv = KFold(n_splits=10, shuffle=True, random_state=1)
#Here we list the parameters we want to tune
space = dict()
#space['n_estimators'] = [1, 20, 100, 500, 1000]
#space['max_depth']=[2, 5, 10, 20]
#space['learning_rate']=[.02, 0.1, 0.2]
space['n_estimators'] = [1000]
#space['n_estimators'] = [10, 20, 100, 500]
space['max_depth']=[2]
#space['min_child_weight']=[1, 5, 10]
#space['learning_rate']=arange(0.1, 4, 0.5)
#space['learning_rate']=[.03, 0.05, .07]
space['learning_rate']=[0.2]
i=1000
r2scoresLR02=[]
#while i<2025:
#    space['n_estimators'] = [i]
    #space['loss']=['linear', 'square', 'exponential']
search = GridSearchCV(LGBMmodel, space, n_jobs=-1, cv=cv, refit=True)
result = search.fit(X_train, y_train)
best_model = result.best_estimator_
yhat_test = best_model.predict(X_test)
yhat_train = best_model.predict(X_train)
yhat_0_2500 = best_model.predict(X_0_2500)#Nu ranges between 0 and 2500 kN
#yhat_2500_5000 = best_model.predict(X_2500_5000)#Nu ranges between 5000 and 7500 kN
#yhat_5000_7500 = best_model.predict(X_5000_7500)#Nu ranges between 5000 and 7500 kN
#yhat_7500Plus = best_model.predict(X_7500Plus)#Nu ranges between 5000 and 7500 kN
print('shape of yhat_test is ',yhat_test.shape)
#xk=[0,10000];yk=[0,10000];
#xk=[0,8000];yk=[0,8000];
#xk=[0,6000];yk=[0,6000];
xk=[0,3000];yk=[0,3000];
#xk=[0,15000];yk=[0,15000];
#yk2=[0,18000];yk3=[0,12000];
#yk2=[0,12000];yk3=[0,8000];
#yk2=[0,9600];yk3=[0,6400];
#yk2=[0,7200];yk3=[0,4800];
yk2=[0,3600];yk3=[0,2400];
pyplot.scatter(yhat_0_2500,df0_2500.Ntest, marker='d',facecolor='midnightblue',edgecolor='midnightblue', label=r'$N_u<2500\hspace{0.5em}LightGBM$')
#pyplot.scatter(yhat_2500_5000,df2500_5000.Ntest, marker='d',facecolor='midnightblue',edgecolor='midnightblue', label=r'$N_u<2500\hspace{0.5em}LightGBM$')
#pyplot.scatter(yhat_5000_7500,df5000_7500.Ntest, marker='d',facecolor='dodgerblue',edgecolor='dodgerblue', label=r'$5000<N_u<7500\hspace{0.5em}LightGBM$')
#pyplot.scatter(yhat_7500Plus,df7500Plus.Ntest, marker='d',facecolor='royalblue',edgecolor='royalblue', label=r'$7500<N_u\hspace{0.5em}LightGBM$')
pyplot.plot(xk,yk,color='black')
pyplot.plot(xk,yk2, dashes=[2,2], color='royalblue')
pyplot.plot(xk,yk3, dashes=[2,2], color='royalblue')
pyplot.ylabel(r'$Measured\hspace{0.5em}N_u\hspace{0.5em}[kN]$')
pyplot.xlabel(r'$LightGBM\hspace{0.5em}predicted\hspace{0.5em} N_u\hspace{0.5em}[kN]$')
pyplot.grid(True)
pyplot.tight_layout()
pyplot.legend()
pyplot.show()
#print('MAPE train:',mean_absolute_percentage_error(y_train, yhat_train))
#print('MAPE test = ',mean_absolute_percentage_error(y_test, yhat_test))
#print('MAPE 0_25 = ',mean_absolute_percentage_error(y_0_25, yhat_0_25))
#print('MAPE 25_50= ',mean_absolute_percentage_error(y_25_50, yhat_25_50))
#print('MAPE 50_100= ',mean_absolute_percentage_error(y_50_100, yhat_50_100))
#print('MAPE 100 Plus= ',mean_absolute_percentage_error(y_100Plus, yhat_100Plus))
print('RMSE test= ', sqrt(mean_squared_error(y_test, yhat_test)))
#print('RMSE 0_25 = ', sqrt(mean_squared_error(y_0_25, yhat_0_25)))
#print('RMSE 25_50 = ', sqrt(mean_squared_error(y_25_50, yhat_25_50)))
#print('RMSE 50_100 = ', sqrt(mean_squared_error(y_50_100, yhat_50_100)))
#print('RMSE 100Plus= ', sqrt(mean_squared_error(y_100Plus, yhat_100Plus)))
print('MAE test = ',mean_absolute_error(y_test, yhat_test))
#print('MAE 0_25 = ',mean_absolute_error(y_0_25, yhat_0_25))
#print('MAE 25_50 = ',mean_absolute_error(y_25_50, yhat_25_50))
#print('MAE 50_100 = ',mean_absolute_error(y_50_100, yhat_50_100))
#print('MAE 100Plus = ',mean_absolute_error(y_100Plus, yhat_100Plus))
#print('R2 train:',r2_score(y_train, yhat_train))
#print('R2 test:',r2_score(y_test, yhat_test))#original
#print('R2 0_25:',r2_score(y_0_25, yhat_0_25))
#print('R2 25_50:',r2_score(y_25_50, yhat_25_50))
#print('R2 50_100:',r2_score(y_50_100, yhat_50_100))
#print('R2 100Plus:',r2_score(y_100Plus, yhat_100Plus))
#print('RMSE test = ',sqrt(mean_squared_error(y_test, yhat_test)))
#print('MAE test = ',mean_absolute_error(y_test, yhat_test))
#print('Best parameters are',search.best_params_)#original
