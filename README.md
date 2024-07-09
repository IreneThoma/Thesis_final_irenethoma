# Thesis Irene Thoma

## overview

This repository provides notebooks and scripts for the masters thesis of Irene Thoma.

It is structured as follows: 
Directories with code: Code-codentbased, Code-ML, Code-preparation
Directories with data: Data_input, Data_modelling, Data_output
Directories with results: EA, figures, pictures

/Code-contentbased
Code-contentbased//content-based.py: The creation of the dashoard

/Code-ML
Code-ML/Featureselection.py: The feature selection
Code-ML/KNN.py: The function of the K-Neirest Neighbour regression
Code-ML/LR.py: The function of the Linear regression, ridge regression, and lasso regression
Code-ML/Random-Linear.py: The statistical comparison between the random regressor and linear regressor
Code-ML/RF.py: The function of the Random Forest regression
Code-ML/SVM.py: The function of the Support Vector Machine regression
Code-ML/SVMlinear.py: The function of the Linear Support Vector regression
Code-ML/SVMSGD.py: The function of the Stochastic Gradient Descent regression

/Code-preparation
Code-preparation/Content_similarity.ipynb: To check what lexical similarity to use with matching ingredients
Code-preparation/Matching_Data.ipynb: To combine the Kaggle and DHLab dataset with each other + check what similarity to use with matching recipe titles
Code-preparation/Split_data_popularity.py: To define the X_train.csv, X_val.csv, X_test.csv, y_train.csv, y_val.csv, y_test.csv datasets
Code-preparation/Trans_DHLabData.ipynb: To translate the DHLab dataset

/Data_input
Data_input/DHLabData.csv: The raw obtained dataset from DHLab
Data_input/RAW_interactions.csv: The raw obtained dataset from Kaggle with ratings
Data_input/RAW_recipes.csv: The raw obtained dataset from Kaggle with recipe information

/Data_modelling
Data_modelling/X_test
Data_modelling/X_train
Data_modelling/X_val
Data_modelling/y_test
Data_modelling/y_train
Data_modelling/y_val

/Data_output
Data_output/DHLdf_cosine.csv: The combined dataset using the cosine similarity
Data_output/DKdf_LinearSVR.csv: The final dataset used in content-based.py
Data_output/translated_DHLabData.csv: The translated DHLabData.csv

/EA: The directory contains the accuracy plot and the residual plot of Random Forest regression and Linear Support Vector regression
/Figures: The results of the data preprocessing and modelling
/Pictures: The images used in the dashboard including the final dashboard pictures