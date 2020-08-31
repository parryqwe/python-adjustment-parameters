# python-adjustment-parameters
## from sklearn.model_selection import train_test_split
1.1 train_test_split(x,y,random_state,train_size)產出train_x,train_y,test_x,test_y  
## from sklearn.model_selection import cross_val_score
1.2 cross_val_score(model,x,y,cv)
## from sklearn.model_selection import RandomizedSearchCV
1.3 RandomizedSearchCV(model, parameters_grid=dict, scoring, cv)
1.3.1 fit(data,label)
1.3.2 bestestimator(屬性)
## from sklearn.model_selection import LeaveOneOut
1.4 cv=LeaveOneOut(len(data))
## from sklearn.model_selection import StratifiedShuffleSplit
1.5 cv=StratifiedShuffleSplit(train_y,n_iter,test_size)
## from sklearn.model_selection import KFold
1.6 KFold(n_splits)
1.6.1 splits(data)產出train_index,test_index
## from sklearn.model_selection import StratifiedKFold
1.7 cv=StratifiedKFold(y,n_folds,random_state)產出k,train_index,test_index

## from sklearn.learning_curve import validation_curve
1.8 validation_curve(estimator,x,y,param_name,param_range=list,cv)
## from sklearn.learning_curve import learning_curve
1.9 learning_curve(estimator,x,y,cv,train_sizes=array)
## from sklearn.model_selection import GridSearchCV 
1.10 GridSearchCV(estimator,param_grid=list(dict) or dict,cv,scoring)
1.10.1 fit
1.10.2 best_estimator_,best_score_,best_params_(屬性)
1.10.3 score
1.10.4 grid_scores_  
## from sklearn.metrics
2.1 roc_auc_score(true,predict)  
2.2 auc(fpr,tpr), roc_curve(true,predict,pos_label)產出fpr,tpr,threshold  
2.3 accuracy_score(true,predict)  
2.4 confusion_matrix(true,predict)  
2.5 mean_squared_error  
2.6 r2_score  
2.7 precision_score,recall_score,f1_score  
2.8 make_scorer(score_func,pos_label,average="micro")  
2.9 classification_report  
## from sklearn.pipeline import make_pipeline
make_pipeline
## from sklearn.pipeline import Pipeline
Pipeline(list(tuple(name,model)))
