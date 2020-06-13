# Data Mining Project
## AUEB | Data Mining from Large Databases and the Web | Semester 7 | 2019 - 2020

The purpose of the project is to create a programming model, which will be able to predict how 
many bicycles will be rented based on statistical data of previous years. More specifically, 
it is given a file (train.csv) for 12165 hours and 14 features for each hour, such as weather, 
month, temperature etc and it is asked to be found the number of bicycles which will be rented 
every hour of the day for new data (test.csv).

Before creating the model, there is a data processing, such as renaming name columns, changing 
type of variables, dropping unnecessary columns and using one-hot encoding technique.

The selected model is a combination of neural networks, MLPRegressor and MLPClassifier, with the 
help of a VotingRegressor. Finding the optimal parameters for the above models helped the GridSearchCV.
