#!/usr/bin/env python
# coding: utf-8

# # Problem Statement

# ## Cross-Sell Prediction
# Your client is an Insurance company that has provided Health Insurance to its customers now they need your help in building a model to predict whether the policyholders (customers) from past year will also be interested in Vehicle Insurance provided by the company.
# 
# An insurance policy is an arrangement by which a company undertakes to provide a guarantee of compensation for specified loss, damage, illness, or death in return for the payment of a specified premium. A premium is a sum of money that the customer needs to pay regularly to an insurance company for this guarantee.
# 
# For example, you may pay a premium of Rs. 5000 each year for a health insurance cover of Rs. 200,000/- so that if, God forbid, you fall ill and need to be hospitalised in that year, the insurance provider company will bear the cost of hospitalisation etc. for upto Rs. 200,000. Now if you are wondering how can company bear such high hospitalisation cost when it charges a premium of only Rs. 5000/-, that is where the concept of probabilities comes in picture. For example, like you, there may be 100 customers who would be paying a premium of Rs. 5000 every year, but only a few of them (say 2-3) would get hospitalised that year and not everyone. This way everyone shares the risk of everyone else.
# 
# Just like medical insurance, there is vehicle insurance where every year customer needs to pay a premium of certain amount to insurance provider company so that in case of unfortunate accident by the vehicle, the insurance provider company will provide a compensation (called ‘sum assured’) to the customer.
# 
# Building a model to predict whether a customer would be interested in Vehicle Insurance is extremely helpful for the company because it can then accordingly plan its communication strategy to reach out to those customers and optimise its business model and revenue. 
# 
# Now, in order to predict, whether the customer would be interested in Vehicle insurance, you have information about demographics (gender, age, region code type), Vehicles (Vehicle Age, Damage), Policy (Premium, sourcing channel) etc.
# 
# ### Understand the data
# ### Variable	Definition
# - **id**: Unique ID for the customer
# - **Gender**:	Gender of the customer
# - **Age**: Age of the customer
# - **Driving_License**: 0 : Customer does not have DL, 1 : Customer already has DL
# - **Region_Code**: Unique code for the region of the customer
# - **Previously_Insured**:	1 : Customer already has Vehicle Insurance, 0 : Customer doesn't have Vehicle Insurance
# - **Vehicle_Age**:	Age of the Vehicle 
# - **Vehicle_Damage**:
# -- **1** : Customer got his/her vehicle damaged in the past.
# -- **0** : Customer didn't get his/her vehicle damaged in the past.
# - **Annual_Premium**:	The amount customer needs to pay as premium in the year
# - **Policy_Sales_Channel**:Policy_Sales_Channel	Anonymised Code for the channel of outreaching to the customer ie. Different Agents, Over Mail, Over Phone, In Person, etc.
# - **Vintage**:	Number of Days, Customer has been associated with the company
# 
# ### target
# - **Response**:	1 :  Customer is interested, 0 : Customer is not interested

# # Data understanding and observations
# #### Import Libraries

# In[135]:


# import libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')


# #### Read the data

# In[136]:


# read the data
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')


# #### Descriptive Analysis

# In[137]:


# total no.of rose and columns
train.shape, test.shape


# In[138]:


#get the datatypes
train.dtypes


# In[139]:


# get all details of dataset
train.info()


# In[140]:


# to see first few rows 
train.head(5)


# In[141]:


# check the missing data elements
train.isna().sum()


# In[142]:


# There are no missing elements in the given dataset


# In[143]:


# to check duplicates
train.duplicated().sum()


# In[144]:


# There is no duplicate values in the given dataset


# In[145]:


#check the details of test dataset
test.info()


# In[146]:


# check target data
train.Response.value_counts()


# In[147]:


train['Response'].value_counts()


# In[148]:


# target in percentage distribution 
train.Response.value_counts(normalize=True)*100


# #### Observations 
# - The dataset contains 381,109 records with 12 columns.
# - Response is the target variable (1 = Interested, 0 = Not Interested).
# - Categorical variables: Gender, Vehicle_Age, Vehicle_Damage.
# - Continuous numerical variables: Age, Annual_Premium, Vintage, Region_Code, etc.
# - No missing values detected.

# In[149]:


train.Response.value_counts().plot(kind='bar')


# - target is imbalanced dataset   

# In[150]:


tgt_col = ['Response']
ign_cols = ['id']


# In[151]:


# statistical info

train.describe().T


# In[152]:


train.drop(columns=ign_cols).describe().T


# In[153]:


train.describe(include='object').T


# In[154]:


train.nunique()


# In[155]:


# check unique values

for col in train.drop(columns=ign_cols).columns:
    print(col,train[col].nunique(),  '=>', train[col].unique())


# In[156]:


sns.distplot(train.Annual_Premium)


# - most of customers paying around 50k annual premium
# - those who have annual premium 100k are very low

# In[157]:


sns.distplot(train.Policy_Sales_Channel)       


# - most of customers through the ploicy sales channel of 150 code

# In[158]:


sns.distplot(train.Vintage)       


# In[159]:


for col in train.select_dtypes(include='object').columns:
    plt.figure(figsize=(5,3))
    sns.countplot(y=train[col])
    plt.show()


# - vehicle damage condition is balanced dataset
# - Age of vehicle is < 2 years
# - Gender is a balanced dataset

# In[160]:


train[['Vintage','Response']].groupby('Vintage').value_counts().plot(kind='barh')


# In[161]:


exp_tgt = train[['Vintage','Response']].groupby('Vintage').value_counts().unstack()
exp_tgt['%'] = exp_tgt[1]/(exp_tgt[0]+exp_tgt[1])*100
exp_tgt.sort_values(exp_tgt.columns[2], ascending=False)


# - Customers who has been associated with the company more than 200 days seeking for the insurance after expiry

# ### Preprocessing

# ##### Steps
# 
# * Address missing data - No missing values found in the given dataset
# * Encoding on category columns
# * Standardize numerical columns
# * Treat data imbalance 

# In[162]:


# import libraries for pre-processing 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc, accuracy_score


# In[163]:


train.dtypes


# In[164]:


# separate category and numeric features
cat_cols = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']

num_cols = ['Age','Annual_Premium','Region_Code','Driving_License','Previously_Insured','Policy_Sales_Channel','Vintage']


# In[165]:


print(tgt_col, ign_cols, cat_cols, num_cols, sep='\n')


# In[166]:


# Encode categorical variables
label_encoders = {}

for col in ['Gender', 'Vehicle_Age', 'Vehicle_Damage']:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    label_encoders[col] = le  # Store encoder for test data transformation


# In[167]:


# Normalize numerical features
scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])


# In[168]:


# Split dataset into train and validation sets
X = train.drop(columns=['id', 'Response'])  # Exclude ID and target column
y = train['Response']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_val.shape, y_train.shape, y_val.shape


# ### LogisticRegression

# In[169]:


# Train a Logistic Regression Model
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Predictions
y_pred = log_model.predict(X_val)
y_pred_proba = log_model.predict_proba(X_val)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_pred_proba)

accuracy, roc_auc


# In[170]:


# Tuning Logistic Regression Model

# set the hyper parameters
params = [
    {
    'model': [LogisticRegression()],
    'model__penalty':['l2',None],
    'model__C':[0.5,3]
    }    
]

# map the grid parameters with pipeline

grid = GridSearchCV(estimator=model_pipeline, param_grid=params, 
                    cv=2, scoring='roc_auc')


# In[171]:


# fit the grid model

grid.fit(train_X, train_y)


# In[172]:


# get the best parameter

grid.best_params_


# In[173]:


# show the grid results

res_df = pd.DataFrame(grid.cv_results_,)
pd.set_option('display.max_colwidth',100)
res_df[['params','mean_test_score','rank_test_score']]


# ### RandomForestClassifier

# In[174]:


from sklearn.ensemble import RandomForestClassifier

# Train a RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_val)
y_pred_proba = rf_model.predict_proba(X_val)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_pred_proba)

accuracy, roc_auc


# ### XGBoost

# In[175]:


import xgboost as xgb

# Define and train XGBoost model
xgb_model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, eval_metric="auc", subsample=1.0, random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_val)
y_pred_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]

# Evaluate the model
accuracy_xgb = accuracy_score(y_val, y_pred_xgb)
roc_auc_xgb = roc_auc_score(y_val, y_pred_proba_xgb)

accuracy_xgb, roc_auc_xgb


# ##### Tried GridSerach with varied parameters; unfortunately system couldn't take this load and restarted multiple times. 

# In[176]:


# Define parameter grid for XGBoost tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 6, 9]    
}

# Tried GridSerach with varied parameters; unfortunately system couldn't take this load and restarted multiple times. 
# Perform GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model, 
    param_grid=param_grid, 
    scoring='roc_auc', 
    cv=3, 
    verbose=1, 
    n_jobs=-1
)

# Run GridSearchCV to find the best parameters
#grid_search.fit(X_train, y_train)

# Get the best parameters and best score
#best_params = grid_search.best_params_
#best_score = grid_search.best_score_

#best_params, best_score ##


# - ***Observations & Model Performance***
# - XGBoost yields the better performance compared with RandomForrestClassifier and Logistic Regression

# #### Pipelining with XGBoost Model

# In[ ]:





# In[177]:


# categorical preperation

cat_pipe_encode = Pipeline(
steps = [
    ('impute_cat', SimpleImputer(strategy='most_frequent')), # missing values
    ('ohe',OneHotEncoder(handle_unknown='ignore')) # categetoy encoding
])


# In[178]:


# numerical features preperation

num_pipe_encode = Pipeline(
steps = [
    ('impute_num', SimpleImputer(strategy='median')), # missing values
    ('scale',StandardScaler()) # standard scaler
])


# In[179]:


# map tranformation to features

preprocess = ColumnTransformer(
    transformers =[
        ('cat_encode',cat_pipe_encode,cat_cols),
        ('num_encode',num_pipe_encode,num_cols)
    ]
)


# In[180]:


# integrate preprocessing and model

model_pipeline = Pipeline(
steps=[
    ('preprocess',preprocess),
    ('model',xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, eval_metric="auc", subsample=1.0, random_state=42))
])


# In[181]:


X = train.drop(columns=ign_cols+tgt_col)
X.head(2)


# In[182]:


y = train[tgt_col]
y.head(2)


# #### train test split

# In[183]:


from sklearn.model_selection import train_test_split, GridSearchCV


# In[184]:


train_X, val_X, train_y, val_y = train_test_split(X,y, 
                                         random_state=42, test_size=0.1)
train_X.shape, val_X.shape, train_y.shape, val_y.shape


# In[185]:


model_pipeline


# In[186]:


# fit the model

model_pipeline.fit(train_X, train_y)


# In[187]:


# predict target with probability

model_pipeline.predict_proba(val_X)


# In[188]:


model_pipeline.predict_proba(val_X)[:,0]


# In[189]:


model_pipeline.predict_proba(val_X)[:,1]


# In[190]:


# predict target 

model_pipeline.predict(val_X)


# In[191]:


# evaluation method

def model_train_val_eval(train_X,val_X,train_y,val_y,model_pipeline):
    
    predicted_train_tgt = model_pipeline.predict(train_X)
    predicted_val_tgt = model_pipeline.predict(val_X)

    print('Train AUC', roc_auc_score(train_y,predicted_train_tgt),sep='\n')
    print('Valid AUC', roc_auc_score(val_y,predicted_val_tgt),sep='\n')

    print('Train cnf_matrix', confusion_matrix(train_y,predicted_train_tgt),sep='\n')
    print('Valid cnf_matrix', confusion_matrix(val_y,predicted_val_tgt),sep='\n')

    print('Train cls_rep', classification_report(train_y,predicted_train_tgt),sep='\n')
    print('Valid cls rep', classification_report(val_y,predicted_val_tgt),sep='\n')

    # plot roc-auc
    y_pred_proba = model_pipeline.predict_proba(val_X)[:,1]
    plt.figure()
    fpr, tpr, thrsh = roc_curve(val_y,y_pred_proba)
    #roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr)
    plt.show()
#model_train_val_eval(train_X,val_X,train_y,val_y,model_pipeline)   


# In[192]:


model_train_val_eval(train_X,val_X,train_y,val_y,model_pipeline)


# ##### Predicting for the test data

# In[193]:


# read the submission file
#predict with the last model
#and upload into the hack website

sub = pd.read_csv('sample_submission_iA3afxn.csv')
sub.head(3)


# In[194]:


test.head(3)


# In[195]:


train.columns.difference(test.columns)


# In[196]:


# updating the existing target values with predicted values
sub['Response'] = model_pipeline.predict(test)


# In[197]:


sub.to_csv('sub_1.csv',index=False)


# In[198]:


sub


# * result uploaded in the analytics vidhya website and the recieved score
# 
# ![image.png](attachment:5ce5f79e-e467-48ca-811e-03048b2a9652.png)

# ### perform oversampling to balance the dataset

# In[209]:


from imblearn.over_sampling import RandomOverSampler


# In[210]:


over_sampling = RandomOverSampler()


# In[211]:


import imblearn
imblearn.__version__


# In[213]:


import sklearn
sklearn.__version__


# In[215]:


train_y.value_counts()


# In[238]:


train_X_os, train_y_os = over_sampling.fit_resample(train_X,train_y)


# In[239]:


train_y_os.value_counts()


# In[240]:


from sklearn.tree import DecisionTreeClassifier


# #### Using GridSearch to hyper tune multi model

# In[241]:


params_2 = [
    {
    'model': [LogisticRegression()],
    'model__penalty':['l2',None],
    'model__C':[0.5,3]
    },
    {
    'model': [DecisionTreeClassifier()],
    'model__max_depth':[3,5]
    }
]


# In[242]:


params_2


# In[243]:


grid_2 = GridSearchCV(estimator=model_pipeline, param_grid=params_2, 
                    cv=2, scoring='roc_auc')


# In[244]:


grid_2.fit(train_X_os, train_y_os)


# In[245]:


grid_2.best_params_


# In[246]:


grid_2.cv_results_


# In[247]:


new_model = grid_2.best_estimator_


# In[248]:


model_train_val_eval(train_X,val_X,train_y,val_y,new_model)


# In[249]:


model_train_val_eval(train_X_os,val_X,train_y_os,val_y,new_model)


# In[250]:


res_df_2 = pd.DataFrame(grid_2.cv_results_,)
pd.set_option('display.max_colwidth',100)
res_df_2[['params','mean_test_score','rank_test_score']]


# In[232]:


# updating the existing target values with predicted values
sub['Response'] = new_model.predict(test)
sub.to_csv('sub_2.csv',index=False)


# * result uploaded in the analytics vidhya website and the recieved score
# 
# ![image_2.png](attachment:7ecbbe3d-8691-4472-b89e-414d5e2ff12b.png)

# #### Using GridSearch to hyper tune multi model along with ensembling

# In[251]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier, StackingClassifier


# In[252]:


params_3 = [
    {
    'model': [LogisticRegression()],
    'model__penalty':['l2',None],
    'model__C':[0.5,3]
    },
    {
    'model': [DecisionTreeClassifier()],
    'model__max_depth':[3,5]
    },
    {
    'model': [StackingClassifier(
    estimators=[
        ['sclf1',RandomForestClassifier()],
        ['sclf2',GradientBoostingClassifier()],
        ['sclf3',AdaBoostClassifier()],],   
        final_estimator=LogisticRegression()

    )],
    'model__sclf1__max_depth':[4,8],
    'model__sclf2__n_estimators':[15,25],    
    'model__sclf3__n_estimators':[5,35],    
    }
]


# In[253]:


grid_3 = GridSearchCV(estimator=model_pipeline, param_grid=params_3, 
                    cv=2, scoring='roc_auc')


# In[ ]:


grid_3.fit(train_X_os, train_y_os)


# In[ ]:





# In[ ]:





# ### Pickling the model

# In[ ]:


import joblib


# In[ ]:


joblib.dump(model_pipeline,'insurance_cross_sell_pred_pipeline_model.pkl')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




