from pandas import read_csv, DataFrame
from numpy import absolute,arange,mean,std,argsort,sqrt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from matplotlib import pyplot 
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score, accuracy_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
pyplot.rc('text',usetex=True)
pyplot.rc('font',size=15)

scaler = MinMaxScaler()
pyplot.rc('text',usetex=True)
pyplot.rc('font',size=15)
colnames = ['B','H','t','L','fy','fc','Ntest']
path='path to inputfile.csv'
df = read_csv(path,header=0,names=colnames)
data=df.values
print('data.shape:',data.shape)
# split into inputs and outputs
#X, y = df.iloc[:, :-1], df.iloc[:, -1]
X, y = data[:, :-1], data[:, -1]
print('X.shape:', X.shape,'y.shape', y.shape)
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

AdaBoostModel = AdaBoostRegressor()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++
cv = KFold(n_splits=10, shuffle=True, random_state=1)
#Here we list the parameters we want to tune
space = dict()
#space['n_estimators'] = arange(1,600,50)
#space['n_estimators'] =[10, 50, 100, 500, 1000, 5000] 
space['n_estimators'] =[100,50, 10] 
space['random_state']=[1,2,3,4]
space['loss']=['linear','square', 'exponential']
#space['learning_rate']=arange(0.1, 4, 0.5)
#space['learning_rate']=[.03, 0.05, .07]
#space['learning_rate']=[2,1,0.1, 0.01, 0.001]
#i=1000
#r2scoresLR02=[]
#while i<2025:
#    space['n_estimators'] = [i]
    #space['loss']=['linear', 'square', 'exponential']
search = GridSearchCV(AdaBoostModel, space, n_jobs=-1, cv=cv, refit=True)
result = search.fit(X_train, y_train)
best_model = result.best_estimator_
yhat_test = best_model.predict(X_test)
yhat_train = best_model.predict(X_train)
#    r2scoresLR02.append(r2_score(y_test, yhat_test))
#    i+=25
#pyplot.scatter(yhat_train,y_train, marker='s',facecolor='seagreen',edgecolor='seagreen', label=r'$Training\hspace{0.5em}set$')#original
#pyplot.scatter(yhat_test,y_test, marker='s',facecolor='none',edgecolor='blue', label=r'$Test\hspace{0.5em}set$')#original
#x=arange(25,625,25)
#x=arange(1000,2025,25)
#pyplot.plot(x,r2scoresLR02, color='blue', label=r'$L_r=0.02$')
#pyplot.plot(x,r2scoresLR05, color='red', dashes=[5,2,2,2], label=r'$L_r=0.05$')
#pyplot.plot(x,r2scoresLR1, color='green', dashes=[5,2], label=r'$L_r=0.1$')
#pyplot.plot(x,r2scoresLR2, color='yellow', dashes=[2,2], label=r'$L_r=0.2$')
#pyplot.xticks(arange(1000,2100,100))
#pyplot.title(r'$max\_depth=5$')
#xk=[0,12500];yk=[0,12500];
#pyplot.plot(xk,yk, color='black')
#pyplot.grid(True)
#pyplot.xlabel(r'$Number\hspace{0.5em}of\hspace{0.5em}trees$')
#pyplot.ylabel(r'$R^2\hspace{0.5em}score$')
#pyplot.legend()
#pyplot.tight_layout()
#pyplot.show()
#R2=r2_score(y_test, yhat)#original
print('MAPE train= ',mean_absolute_percentage_error(y_train, yhat_train))
print('RMSE train= ',sqrt(mean_squared_error(y_train, yhat_train)))
print('MAE train= ',mean_absolute_error(y_train, yhat_train))
print('R2 train:',r2_score(y_train, yhat_train))
print('MAPE test= ',mean_absolute_percentage_error(y_test, yhat_test))
print('RMSE test= ',sqrt(mean_squared_error(y_test, yhat_test)))
print('MAE test= ',mean_absolute_error(y_test, yhat_test))
print('R2 test:',r2_score(y_test, yhat_test))#original
print('Best parameters are',search.best_params_)#original
