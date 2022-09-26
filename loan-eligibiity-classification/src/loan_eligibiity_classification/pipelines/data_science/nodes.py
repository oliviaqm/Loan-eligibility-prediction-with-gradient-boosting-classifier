import logging
from symbol import parameters
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.preprocessing import LabelBinarizer,StandardScaler,OrdinalEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.stats import boxcox
from sklearn.linear_model import LogisticRegression,RidgeClassifier, PassiveAggressiveClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import joblib

from kedro.extras.datasets.pickle import PickleDataSet

def binarize_target_var(data: pd.DataFrame) -> pd.DataFrame:
    """Encode the target variable into zeroes and ones.

    Args:
        data: original dataset containing features and target.
        
    Returns:
        Binarized target variable in one-dimensional structure.
    """
    lb_style = LabelBinarizer()
    lb_results = lb_style.fit_transform(data['Loan Status'])
    #print(lb_results) # lb_results is a multidimensional array.
    y=lb_results
    y=y.ravel() # use ravel to flatten lb_results into a 1-d structure, like a list.
    #check last 10 rows of y
    #print(y[-10:])
    #check last 10 rows of preprocessed_data_4
    #print(data['Loan Status'][-10:])    
    return y

def feature_scaling(preprocessed_data_4: pd.DataFrame) -> pd.DataFrame:
    """
    Bring all independent variables to the same scale.

    Args:
        data: preprocessed dataset containing features.
        
    Returns:
        Preprocessed dataframe containing scaled features.
    """
    X_scaled = preprocessing.scale(preprocessed_data_4)
    print(X_scaled.shape)
    return X_scaled

def split_data(scaled_features, y, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        scaled_features: Scaled features.
        y: Binarized target var.
        parameters: Parameters defined in parameters/data_science.yml. 
    Returns:
        Split data.
    """
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, y, test_size=parameters["test_size"], random_state=parameters["random_state"])
    return X_train, X_test, y_train, y_test

def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: Dict) -> GradientBoostingClassifier:
    """Trains the Gradient Boosting Machine model. 

    Args:
        X_train: Training data of independent features.
        y_train: Training data for target var (0: loan granted, 1: loan refused).

    Returns:
        Trained model.
    """
    gbm=GradientBoostingClassifier(max_depth= parameters["max_depth"], n_estimators=parameters["n_estimators"], max_features = parameters["max_features"])
    gbm.fit(X_train, y_train)
    return gbm

def evaluate_models(X_train, y_train, X_test, y_test, parameters: Dict, model_type = "non-balanced"):
    clfs = {'GradientBoosting': GradientBoostingClassifier(max_depth= parameters["max_depth"], n_estimators=parameters["n_estimators"], max_features = parameters["max_features"]),
            'LogisticRegression' : LogisticRegression(),
            #'GaussianNB': GaussianNB(),
            'RandomForestClassifier': RandomForestClassifier(n_estimators=10),
            'XGBClassifier': XGBClassifier()
            }
    cols = ['model','matthews_corrcoef', 'roc_auc_score', 'precision_score', 'recall_score','f1_score']

    models_report = pd.DataFrame(columns = cols)
    conf_matrix = dict()

    for clf, clf_name in zip(clfs.values(), clfs.keys()):

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:,1]

        print('computing {} - {} '.format(clf_name, model_type))

        tmp = pd.Series({'model_type': model_type,
                         'model': clf_name,
                         'roc_auc_score' : metrics.roc_auc_score(y_test, y_pred_proba),
                         'matthews_corrcoef': metrics.matthews_corrcoef(y_test, y_pred), # Matthews Correlation Coefficient is a good metric for imbalanced class problems
                         'precision_score': metrics.precision_score(y_test, y_pred),
                         'recall_score': metrics.recall_score(y_test, y_pred),
                         'f1_score': metrics.f1_score(y_test, y_pred)})
        # Construct a pandas df models report, with the keys in tmp as columns and the values in tmp.
        models_report = models_report.append(tmp, ignore_index = True)
        # Construct a dictionary-form confusion matrix with y_test (actual) values as indexes and y_pred (predicted) value as columns.
        conf_matrix[clf_name] = pd.crosstab(y_test, y_pred, rownames=['True'], colnames= ['Predicted'], margins=False)
        print(conf_matrix)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba, drop_intermediate = False, pos_label = 1) # drop_intermediate: Whether to drop some suboptimal thresholds which would not appear on a plotted ROC curve. This is useful in order to create lighter ROC curves.

        plt.figure(1, figsize=(6,6))
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title('ROC curve - {}'.format(model_type))
        plt.plot(fpr, tpr, label = clf_name )
        plt.legend(loc=2, prop={'size':11})
        plt.plot([0,1],[0,1], color = 'black')
    
    return models_report, pd.DataFrame(y_pred), pd.DataFrame(y_pred_proba)
