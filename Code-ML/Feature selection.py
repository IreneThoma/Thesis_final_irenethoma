import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

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

#Feature selection
label_encoder = LabelEncoder()
X_train_encoded = X_train.copy()
X_train_encoded['Recipename'] = label_encoder.fit_transform(X_train['Recipename'])
X_train_encoded['Ingredients'] = label_encoder.fit_transform(X_train['Ingredients'])
X_train_encoded['Steps'] = label_encoder.fit_transform(X_train['Steps'])
mi_scores = mutual_info_regression(X_train_encoded, y_train.values.ravel())
mi_scores = pd.Series(mi_scores, index=X_train_encoded.columns)
mi_scores = mi_scores.sort_values(ascending=False)
plt.figure(figsize=(20, 40))
mi_scores.plot(kind='bar', color='deepskyblue', edgecolor='black')
plt.title('Mutual Information Scores for Features', fontsize=20)  
plt.xlabel('Features', fontsize=20)
plt.ylabel('Mutual Information', fontsize=20) 
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()