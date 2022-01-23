from pandas import read_csv
from numpy import array
url = 'http://23.94.143.238/ml_data/stub.csv'
sutunlar = [r'$b$', r'$h$', r'$t$', r'$l$', r'$f_y$', r'$f_c$', r'$N$']
vericercevesi = read_csv(url, names=sutunlar)
dizimiz = vericercevesi.values
X = dizimiz[:,0:6]
y = dizimiz[:,6]

X=vericercevesi[[r'$b$',r'$h$',r'$t$', r'$l$', r'$f_y$', r'$f_c$']]
y=vericercevesi[[r'$N$']]
import xgboost
import shap
from matplotlib import pyplot
pyplot.rc('text',usetex=True)
pyplot.rc('font',size=25)
pyplot.rcParams.update({'font.size': 25})
# train an XGBoost model
model = xgboost.XGBRegressor().fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(X)

shap.initjs()
#shap.plots.force(shap_values[0], matplotlib=True, show=False)
#pyplot.xlabel(r'$b$')
#pyplot.ylabel(r'$SHAP \hspace{0.5em}value\hspace{0.5em} for$')
#shap.plots.beeswarm(shap_values, show=False,color_bar_label=r'$Feature\hspace{0.5em} value$' )
shap.plots.scatter(shap_values[:,r'$h$'],show=False, color=shap_values)#color_bar_labels
fig, ax = pyplot.gcf(), pyplot.gca()
#ax.set_xlabel(r'$SHAP\hspace{0.5em} value$', fontdict={"size":15})
ax.set_ylabel(r'$SHAP \hspace{0.5em}value\hspace{0.5em} for\hspace{0.5em}h$', fontdict={"size":25})
ax.set_yticks(array([-300,-200,-100,0,100,200, 300, 400,500]))
#ax.set_xticks(array([0,250,500,750,1000,1250,1500,1750,2000]))
ax.tick_params(axis='y', labelsize=25)
ax.tick_params(axis='x', labelsize=25)
ax.set_xlabel(r'$h$',fontdict={"size":25})
pyplot.tight_layout()
