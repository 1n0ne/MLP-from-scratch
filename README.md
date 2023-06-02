This an MLP(Multilayer perceptron) implementation from scratch, the MLP model goal is to predict whether breast 
cancer is benign or 
malignant, the model is traind on the 
Wisconsin Diagnostic Breast Cancer (WDBC) dataset from the UCI machine learning repository which available in the link below:
https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)
to run and test the code u must install python 3 and the follwing library:
-numpy
-pandas
-matplotlib
-scikit-learn
then load the dataset and pass it's file to the data_processing function.
u also can find the complete model on Completed_model.joblib and load it via the command joblib.load("Completed_model.joblib") then u can get 
the model prediction via the predict function