from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


diabetes=load_diabetes()
X=diabetes.data
y=diabetes.target



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
elasticnet=ElasticNet()
elasticnet_params_grid={"alpha":[0.1,1,10,100],"l1_ratio":[0.1,0.5,0.7,0.9]}
elasticnet_grid_search=GridSearchCV(elasticnet,elasticnet_params_grid,cv=5,scoring="neg_mean_squared_error")
elasticnet_grid_search.fit(X_train,y_train)
print("ElasticNet Regression Best Params: ",elasticnet_grid_search.best_params_)
print("ElasticNet Regression Best Score: ",elasticnet_grid_search.best_score_)
best_elasticnet_model=elasticnet_grid_search.best_estimator_
y_pred_elasticnet=best_elasticnet_model.predict(X_test)
elasticnet_mse=mean_squared_error(y_test,y_pred_elasticnet)
print("ElasticNet Regression MSE: ",elasticnet_mse)
