## Credit Card Fraud ##
This repository showcases a machine learning project for detecting fraudulent transactions using the credit card fraud dataset from Kaggle. The dataset was gathered by Worldline and ULB's Machine Learning Group and can be found [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

### Data Analysis ###
Exploratory Data Analysis (EDA) was performed to understand the dataset's imbalance and feature distribution, using Matplotlib for visualization. Insights from this phase guided the data preprocessing and modeling approach.

### Model Development ###
Two models were chosen: Random Forest, and XGBoost implementation of Gradient Boosted Trees. In both cases,grid search was used to tune three hyperparameters for optimal performance. For the random forest, the hyperparameters were: the number of estimators, maximum tree depth, and minimum decrease in impurity required to make a split. For the XGBoost model, the hyperparameters selected were learning rate, maximum tree depth, and minimum decrease in impurity required to make a split.

### Model Evaluation ###
The data was split into training, validation, and testing using stratified splits to preserve the ratios between classes. Furthermore, to address the extreme imbalance in the data, all trained models were weight-balanced classifiers. In each iteration of the grid search loop, a model was trained on the training data with selected hyperparameters, then evaluated on the validation dataset based on F1, and finally evaluated on the test data using AUROC.

### Results ###
The results show a promising 0.96 test set AUROC for the optimal random forest model and a test set AUROC of 0.98 for the optimal XGBoost model.
