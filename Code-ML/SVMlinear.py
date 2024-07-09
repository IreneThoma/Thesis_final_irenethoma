import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.svm import LinearSVR
import time
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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

#LinearSVR regression
def Linearsvr (X_train_combined, y_train, X_val_combined, y_val):
    param_grid = {
        'C': [10, 20, 30, 40],       
        'epsilon': [5, 10, 15],   
        'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']
    }
    svm_model = LinearSVR()
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

    return y_pred, best_svm_model

def accuracy_plot(y_val, y_pred):
    y_val = np.asarray(y_val).flatten()
    y_pred = np.asarray(y_pred).flatten()

    xy = np.vstack([y_val, y_pred])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    y_val, y_pred, z = y_val[idx], y_pred[idx], z[idx]

    plt.scatter(y_val, y_pred, c=z, s=10, cmap='viridis') 
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.colorbar(label='Density')
    plt.show()

def residual_plot(y_val, y_pred):
    y_val = np.asarray(y_val).flatten()
    y_pred = np.asarray(y_pred).flatten()

    xy = np.vstack([y_val, y_val - y_pred])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    y_val, residuals, z = y_val[idx], (y_val - y_pred)[idx], z[idx]

    plt.scatter(y_val, residuals, c=z, s=10, cmap='viridis') 
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Actual')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.colorbar(label='Density')
    plt.show()

# print time
start_time = time.time()
y_pred, best_svm_model = Linearsvr(X_train_combined, y_train, X_val_combined, y_val)
end_time = time.time()
time_taken = end_time - start_time
print("Time taken:", time_taken)
print(best_svm_model)

#accuracy_plot(y_val, y_pred)
residual_plot(y_val, y_pred)

###############################################################
#save model
predicted_df = X_val_incl.copy()
predicted_df['Predicted_Score'] = y_pred
predicted_df.to_csv('Data_output/DKdf_LinearSVR.csv', index=False)