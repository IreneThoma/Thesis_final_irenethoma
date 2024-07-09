import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

#Load data
DKdf_first = pd.read_csv('Data_output/DHLdf_cosine.csv', delimiter=',')
DKdf_first.iloc[:, 9:22] = DKdf_first.iloc[:, 9:22].abs() #make the sustainability scores absolute ('-150 ml milk')

#plot frequency data
def histogram(columnname):
    hist, bins = np.histogram(DKdf_first[columnname], bins=int(max(DKdf_first[columnname])), range=(0,max(DKdf_first[columnname])))
    plt.bar(bins[:-1], hist, width=1, color='skyblue', edgecolor='black') 
    plt.xlabel(columnname)
    plt.ylabel('Frequency')
    plt.title('Frequency of columnname totalghge/portion (kg CO2eq)')
    plt.show()

#print(max(DKdf_first['totalghge/portion (kg CO2eq)']))
#histogram('totalghge/portion (kg CO2eq)')

#remove outliers
columns = [ 'totalghge/portion (kg CO2eq)', 
            'totallanduse/portion (m2)', 
            'totalfreshwaterwithdrawals/portion (L)', 
            'totalstressweightedwateruse/portion (L)', 
            'totaleutrophyingemissions/portion (g PO43-eq)',
            'totalcalories/portion (J)']
Q1 = DKdf_first[columns].quantile(0.25)
Q3 = DKdf_first[columns].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_any = (DKdf_first[columns] < lower_bound) | (DKdf_first[columns] > upper_bound)
outliers_row = outliers_any.any(axis=1)
DKdf_first = DKdf_first[~outliers_row]

print(max(DKdf_first['totalghge/portion (kg CO2eq)']))
histogram('totalghge/portion (kg CO2eq)')

#scale columns from 0 till 100
columns_to_remove = ['totalghge/portion (kg CO2eq)', 
                    'totallanduse/portion (m2)', 
                    'totalfreshwaterwithdrawals/portion (L)', 
                    'totalstressweightedwateruse/portion (L)', 
                    'totaleutrophyingemissions/portion (g PO43-eq)',
                    'totalcalories/portion (J)',
                    'rating'
                    ]
columns_to_scale = ['totalghge/portion (kg CO2eq)', 
                    'totallanduse/portion (m2)', 
                    'totalfreshwaterwithdrawals/portion (L)', 
                    'totalstressweightedwateruse/portion (L)', 
                    'totaleutrophyingemissions/portion (g PO43-eq)',
                    'rating'
                    ]
columns_to_transform = ['totalghge/portion (kg CO2eq)', 
                    'totallanduse/portion (m2)', 
                    'totalfreshwaterwithdrawals/portion (L)', 
                    'totalstressweightedwateruse/portion (L)', 
                    'totaleutrophyingemissions/portion (g PO43-eq)']

DKdf_first = DKdf_first[(DKdf_first[columns_to_remove] != 0).all(axis=1)]       #remove 57,495 rows where minimal one sustainability score is 0
scaler = MinMaxScaler(feature_range=(0, 100))                                   # scale sustainability scores between 0-100
DKdf_first[columns_to_scale] = scaler.fit_transform(DKdf_first[columns_to_scale])
DKdf_first[columns_to_transform] = DKdf_first[columns_to_transform] * -1 + 100        # 'rating' doesn't have to be transformed 


weights = [5.3 * 10**-10, 8.9 * 10**-9, 6.0 * 10**-13, 0, 6.1 * 10**-7, 1.2 * 10**-7]
columns_to_sum = ['totalghge/portion (kg CO2eq)', 'totallanduse/portion (m2)', 'totalfreshwaterwithdrawals/portion (L)', 'totalstressweightedwateruse/portion (L)', 'totaleutrophyingemissions/portion (g PO43-eq)', 'rating']

#Scale the weights to range from 0 to 1
min_weight = min(weights)
max_weight = max(weights)
scaled_weights = [(weight - min_weight) / (max_weight - min_weight) for weight in weights]

for i, column_name in enumerate(columns_to_sum):
    DKdf_first[column_name] = DKdf_first[column_name] * scaled_weights[i]

#columns_to_sum = ['totaleutrophyingemissions/portion (g PO43-eq)'] #Use this to check the test with one factor
DKdf_first['Score'] = DKdf_first[columns_to_sum].sum(axis=1)


#try to sample data, because of the runntime
#DKdf = DKdf_first.sample(n=10000, replace=False)
DKdf = DKdf_first
#DKdf = DKdf_first[~DKdf_first['documentname'].str.contains('kochbar')].copy() #remarkably, MSE is way lower when using also kochbar titles, meaning that they have some addition to it

#Scale features
scale_features = ['cookingtimelist', 'steps_number', 'n_ingredients', 'totalcalories/portion (J)']
DKdf[scale_features] = StandardScaler().fit_transform(DKdf[scale_features])

#Split data
new_feature_names = {
    'title': 'Recipename',
    'vegetarianrecipe': 'Vegetarian',
    'veganrecipe': 'Vegan',
    'cookingtimelist': 'Cookingtime',
    'steps_number': 'Numberofsteps',
    'n_ingredients': 'Numberofingredients',
    'ingredientlist': 'Ingredients',
    'cookingmethod': 'Steps',
    'totalcalories/portion (J)': 'Totalcalories/portion(J)'
}
DKdf.rename(columns=new_feature_names, inplace=True)

#features = ['Recipename', 'url', 'Vegetarian', 'Vegan','Cookingtime', 'Numberofsteps', 'Numberofingredients', 'Ingredients', 'Steps', 'Totalcalories/portion(J)']
features = ['url', 'Vegetarian', 'Vegan', 'Recipename', 'Steps', 'Ingredients', 'Cookingtime']


X = DKdf[features]
y = DKdf['Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

#Save data
X_train.to_csv('Data_modelling/X_train.csv', index=False)
X_val.to_csv('Data_modelling/X_val.csv', index=False)
X_test.to_csv('Data_modelling/X_test.csv', index=False)
y_train.to_csv('Data_modelling/y_train.csv', index=False)
y_val.to_csv('Data_modelling/y_val.csv', index=False)
y_test.to_csv('Data_modelling/y_test.csv', index=False)