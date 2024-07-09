import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kstest

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

#Linear Regression
def Linear_r(X_train_combined, y_train, X_val_combined, y_val):
    model = LinearRegression()
    model.fit(X_train_combined, y_train)
    y_pred = model.predict(X_val_combined)
    MAE = mean_absolute_error(y_val, y_pred)
    print("MAE Linear:", MAE)
    return np.squeeze(y_pred)

#Randon Regression
def RandomRegressor(X_train_combined, y_train, X_val_combined, y_val):
    dummy_regressor = DummyRegressor()
    dummy_regressor.fit(X_train_combined, y_train)
    y_pred = dummy_regressor.predict(X_val_combined)
    MAE = mean_absolute_error(y_val, y_pred)
    print("MAE Random:", MAE)
    return y_pred

#Load functions
linear_model = Linear_r(X_train_combined, y_train, X_val_combined, y_val)
random_model = RandomRegressor(X_train_combined, y_train, X_val_combined, y_val)

#############################################################################################
#CHECK FOR NORMALITY
linear_model_sample = np.random.choice(linear_model, 200)

#Plot histogram with Shapiro-Wilk test result
def plot_histogram_with_test(sample, model_name):
    plt.figure(figsize=(8, 6))
    plt.hist(sample, bins=20, density=True, alpha=0.6, color='deepskyblue', edgecolor='black')
    plt.title(f'{model_name} Model Sample Distribution')
    plt.xlabel('Values')
    plt.ylabel('Frequency')

    #Perform Kolmogorov-Smirnov test
    stat, p = kstest(sample, 'norm')
    plt.text(0.05, 0.95, f'Kolmogorov-Smirnov test {model_name}: p-value: {p:.2e}', transform=plt.gca().transAxes)
    if p < 0.05:
        plt.text(0.05, 0.90, 'Not normally distributed', transform=plt.gca().transAxes, color='r')
    else:
        plt.text(0.05, 0.90, 'Normally distributed', transform=plt.gca().transAxes, color='b')

    plt.show()
 
plot_histogram_with_test(linear_model_sample, 'linear')
#So not normally distributed so Wilcoxon signed-rank test statistic

####################################################################################################
y_val_array = np.array(y_val['Score'])
linear_residuals = abs(y_val_array - linear_model)
random_residuals = abs(y_val_array - random_model)

d = linear_residuals - random_residuals
d_sample = d[:1000]

#perform Wilcoxon signed-rank tests
def wilcoxon_test(residuals1, model_name):
    stat, p = stats.wilcoxon(residuals1, alternative = 'less')
    print(f'Wilcoxon signed-rank test for {model_name}: p-value: {p}')
    if p < 0.05:
        print("Linear is significately lower.")
    else:
        print("There is no significant difference.")
        
def t_test(residuals1, model_name):
    stat, p = stats.ttest_1samp(residuals1, 0)
    print(f'T-test for {model_name}: p-value: {p}')
    if p < 0.05:
        print("Linear is significately lower.")
    else:
        print("There is no significant difference.")

wilcoxon_test(d_sample, 'linear') # p-value: 0.0015130985726498228

def acuracy_plot(y_val, y_pred):
    y_val = np.asarray(y_val).flatten()
    y_pred = np.asarray(y_pred).flatten()

    plt.scatter(y_val, y_pred, s=10, cmap='viridis') 
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.show()

#acuracy_plot(y_val, random_model)