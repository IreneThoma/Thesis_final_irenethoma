import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import classification_report
import time
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

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

#KNN function
def KNN_mae(X_train_combined, y_train, X_val_combined, y_val, max_neighbors=30):
    mae_values = []
    mse_values = []
    rmse_values = []
    for n_neighbors in range(1, max_neighbors+1):
        print(n_neighbors)
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
        model.fit(X_train_combined, y_train)
        y_pred = model.predict(X_val_combined)
        MAE = mean_absolute_error(y_val, y_pred)
        MSE = mean_squared_error(y_val, y_pred)
        RMSE = np.sqrt(MSE)
        mae_values.append(MAE)
        mse_values.append(MSE)
        rmse_values.append(RMSE)
        print("MAE for {} neighbors: {}".format(n_neighbors, MAE))
        print("MSE for {} neighbors: {}".format(n_neighbors, MSE))
        print("RMSE for {} neighbors: {}".format(n_neighbors, RMSE))
    
    return y_pred, mae_values

def acuracy_plot(y_val, y_pred):
    #Plot predicted vs actual
    plt.scatter(y_val, y_pred, color='blue')
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=4)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.show()

#Print time
start_time = time.time()
knn_model, mae_values = KNN_mae(X_train_combined, y_train, X_val_combined, y_val)
end_time = time.time()
time_taken = end_time - start_time
print("Time taken:", time_taken)
acuracy_plot(y_val, knn_model)



#Plot the MAE values vs the number of neighbors
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(mae_values) + 1), mae_values, marker='o', linestyle='-')
plt.title('MAE vs. Number of Neighbors in KNN')
plt.xlabel('Number of Neighbors')
plt.ylabel('Mean Absolute Error (MAE)')
plt.grid(True)
#plt.show()
plt.savefig('KNN_all_part.png')


