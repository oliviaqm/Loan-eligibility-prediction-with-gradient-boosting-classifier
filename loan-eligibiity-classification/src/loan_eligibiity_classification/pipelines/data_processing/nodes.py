from fancyimpute import KNN, SoftImpute
import pandas as pd
import pandas as pd
import numpy as np
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
from matplotlib import pyplot
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import joblib
from imblearn.over_sampling import SMOTE

# def create_small_dataset(data:pd.DataFrame) ->pd.DataFrame:
#     """
#     Create a small dataset to work with, in case of problems with computer memory.
#     """
#     small_data = data.sample(frac=0.3)
#     return small_data

def replace_outliers(data:pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess data by replacing outliers.
    Args:
        loans raw data
    Returns:
        Semi-processed data, with outliers replaced.
    """
    # Current Loan Amount
    temp = np.array(data["Current Loan Amount"].values.tolist())
    Q2 = data["Current Loan Amount"].quantile(0.5)
    data["Current Loan Amount"] = np.where(temp > 9999998, Q2, temp).tolist()   
    
    # "Credit Score" has values exceeding 800. Divide these values by 10.
    data["Credit Score"]=np.where(data["Credit Score"]>800, data["Credit Score"]/10, data["Credit Score"])
   
    # "Annual Income": replace high outliers with 99% quantile.
    inc_threshold = data['Annual Income'].quantile(0.99)
    data.loc[data['Annual Income'] > inc_threshold, 'Annual Income'] = inc_threshold
    
    # "Monthly Debt": Remove '$' sign and convert to numerical form, then remove outliers.
    data['Monthly Debt']=data['Monthly Debt'].str.replace('$', '', regex=True)
    data['Monthly Debt']=pd.to_numeric(data['Monthly Debt'] )
    debt_threshold = data['Monthly Debt'].quantile(0.999)
    data.loc[data['Monthly Debt'] > debt_threshold, 'Monthly Debt'] = debt_threshold
    
    # "Number of Open Accounts"
    num_acc_threshold = data['Number of Open Accounts'].quantile(0.999)
    data.loc[data['Number of Open Accounts'] > num_acc_threshold, 'Number of Open Accounts'] = num_acc_threshold

    # "Current Credit Balance"
    credit_bal_threshold = data['Current Credit Balance'].quantile(0.99)
    data.loc[data['Current Credit Balance'] > credit_bal_threshold, 'Current Credit Balance'] =credit_bal_threshold
    # log-transform this feature to normalize its distribution
    data['Current Credit Balance']=data['Current Credit Balance']**(1/2)

    # "Maximum Open Credit"
    # replace '#VALUE!' with NaN
    data['Maximum Open Credit']=data['Maximum Open Credit'].replace('#VALUE!', np.nan, regex=True)
    data['Maximum Open Credit']=pd.to_numeric(data['Maximum Open Credit'])
    max_open_credit_threshold = data['Maximum Open Credit'].quantile(0.99)
    data.loc[data['Maximum Open Credit'] > max_open_credit_threshold, 'Maximum Open Credit'] = max_open_credit_threshold

    return data

def standardize_class_labels(data:pd.DataFrame) -> pd.DataFrame:
    """
    "Home Ownership" consists of 2 classes, 'Home Mortgage' and 'haveMortgage'.
    These 2 classes should be standardized into 1.
    "Purpose" also has 2 classes 'Other' and 'other' that need to be combined into 1.
    Args: 
        loans raw data
    Returns:
        Semi-processed data, with class labels of categorical features standardized.
    """
    data['Home Ownership']=data['Home Ownership'].str.replace('HaveMortgage', 'Home Mortgage', regex=True)
    data['Purpose']=data['Purpose'].str.replace('Other', 'other', regex=True)
    
    return data

def convert_cat_to_num(data:pd.DataFrame)-> pd.DataFrame: 
    """
    Convert categorical features into numerical form, using get_dummies.
    """
    cat_cols = ['Term','Years in current job','Home Ownership','Purpose']

    for c in cat_cols:
        data[c] = pd.factorize(data[c])[0]
    
    return data

def replace_missing_values(data:pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw data, by replacing missing values.
    Args:
        loans Raw data.
    Returns:
        Semi-preprocessed loans data, with missing values filled. 
 
    """
    # Current Loan Amount: missing values are recorded as 9999998
    temp = np.array(data["Current Loan Amount"].values.tolist())
    Q2 = data["Current Loan Amount"].quantile(0.5)
    data["Current Loan Amount"] = np.where(temp > 9999998, Q2, temp).tolist()   
    
    # Impute missing values, using SoftImpute. 
    updated_data=pd.DataFrame(data=SoftImpute().fit_transform(data[data.columns[3:19]],), columns=data[data.columns[3:19]].columns, index=data.index)
    
    return updated_data


