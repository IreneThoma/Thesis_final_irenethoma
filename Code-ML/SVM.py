import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import time
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

#Load data
X_train_incl = pd.read_csv('Data_modelling/X_train.csv', delimiter=',')
X_val_incl = pd.read_csv('Data_modelling/X_val.csv', delimiter=',')
X_test_incl = pd.read_csv('Data_modelling/X_test.csv', delimiter=',')
y_train = pd.read_csv('Data_modelling/y_train.csv', delimiter=',')
y_val = pd.read_csv('Data_modelling/y_val.csv', delimiter=',')
y_test = pd.read_csv('Data_modelling/y_test.csv', delimiter=',')

X_train = X_train_incl.drop(columns=['url', 'Vegetarian', 'Vegan'])
X_val = X_val_incl.drop(columns=['url', 'Vegetarian', 'Vegan'])
X_test = X_test_incl.drop(columns=['url', 'Vegetarian', 'Vegan'])

X_train['Steps'].fillna('', inplace=True)
X_val['Steps'].fillna('', inplace=True)
X_test['Steps'].fillna('', inplace=True)

#Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_text_tfidf = vectorizer.fit_transform(X_train['Recipename'] + ' ' + X_train['Ingredients']+ ' ' + X_train['Steps'])
X_val_text_tfidf = vectorizer.transform(X_val['Recipename'] + ' ' + X_val['Ingredients']+ ' ' + X_val['Steps'])
X_test_text_tfidf = vectorizer.transform(X_test['Recipename'] + ' ' + X_test['Ingredients']+ ' ' + X_test['Steps'])
X_train_combined = hstack([X_train_text_tfidf, X_train.drop(columns=['Recipename', 'Ingredients', 'Steps'])])
X_val_combined = hstack([X_val_text_tfidf, X_val.drop(columns=['Recipename', 'Ingredients', 'Steps'])])
X_test_combined = hstack([X_test_text_tfidf, X_test.drop(columns=['Recipename', 'Ingredients', 'Steps'])])

#Support Vector Machine regressor
def SVM(X_train_combined, y_train, X_val_combined, y_val):
    param_grid = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [45, 50, 55, 60, 65],
        'epsilon': [1, 5, 10, 15],
        'gamma': ['scale', 'auto'],
    }
    svm_model = SVR()
    grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=2, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_combined, y_train.values.ravel()) 
    best_svm_model = grid_search.best_estimator_
    y_pred = best_svm_model.predict(X_val_combined)
    
    MAE = mean_absolute_error(y_val.values.ravel(), y_pred)
    MSE = mean_squared_error(y_val.values.ravel(), y_pred)
    RMSE = np.sqrt(MSE)
    print("MAE SGD SVM:", MAE)
    print("MSE SGD SVM:", MSE)
    print("RMSE SGD SVM:", RMSE) 

    return best_svm_model

#print time
start_time = time.time()
best_svm_model = SVM(X_train_combined, y_train, X_val_combined, y_val)
print(best_svm_model)
end_time = time.time()
time_taken = end_time - start_time
print("Time taken:", time_taken)
