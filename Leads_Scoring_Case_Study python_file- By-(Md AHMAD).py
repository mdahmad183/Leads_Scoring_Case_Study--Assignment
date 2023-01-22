#!/usr/bin/env python
# coding: utf-8

# # Leads Score - Case_Study 

# ## Problem Statement
# 
# An X Education has appointed you to help them select the most promising leads, i.e. the leads that are most likely to convert into paying customers. The company requires you to build a model wherein you need to assign a lead score to each of the leads such that the customers with a higher lead score have a higher conversion chance and the customers with a lower lead score have a lower conversion chance. The CEO, in particular, has given a ballpark of the target lead conversion rate to be around 80%.

# ## Data
# 
# You have been provided with a leads dataset from the past with around 9000 data points. This dataset consists of various attributes such as Lead Source, Total Time Spent on Website, Total Visits, Last Activity, etc. which may or may not be useful in ultimately deciding whether a lead will be converted or not. The target variable, in this case, is the column ‘Converted’..........

# ## Goals of the Case Study
# 
# * There are quite a few goals for this case study:
# 
#   * Build a logistic regression model to assign a lead score between 0 and 100 to each of the leads which can be used by the company to target potential leads. A higher score would mean that the lead is hot, i.e. is most likely to convert whereas a lower score would mean that the lead is cold and will mostly not get converted.
#   
#   * There are some more problems presented by the company which your model should be able to adjust to if the company's requirement changes in the future so you will need to handle these as well. These problems are provided in a separate doc file. Please fill it based on the logistic regression model you got in the first step. Also, make sure you include this in your final PPT where you'll make recommendations.
#  

# # Importing 
# # required packages &
# 
# # Supress Warnings

# In[601]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



import warnings
warnings.filterwarnings('ignore')


# # A : Loading and Cleaning Data
# ## A.1 : Import Data

# In[602]:


LS = pd.read_csv("Leads.csv")
LS.head()


# ## A.2 : get shape, info & Describe the dataframe
# 
# ### This helps to give a good idea of the dataframes.

# In[603]:


LS.shape


# In[604]:


LS.info()


# In[605]:


# already we show non_null is present in whole data_set
# but specily we have a function to know null values .

LS.isnull().sum()


# In[606]:


round(100*(LS.isnull().sum()/len(LS.index)), 2)


# In[607]:


LS.describe()


# #  A.3 : Cleaning the dataframe

# In[608]:


# 1st Converting all the values to lower case
LS = LS.applymap(lambda s:s.lower() if type(s) == str else s)

# Replacing 'Select' with NaN (Since it means no option is selected)
LS = LS.replace('select',np.nan)

# Checking if there are columns with one unique value since it won't affect our analysis
LS.nunique()


# In[609]:


# ACCORDING TO INFO OF DATA WE CLEARLY DISTRIBUTE THE NUMARICAL AND CATEGORICAL VARIABLES .
LS.columns


# In[610]:


# Dropping unique valued columns
LS_1= LS.drop(['Magazine','Receive More Updates About Our Courses','I agree to pay the amount through cheque','Get updates on DM Content','Update me on Supply Chain Content'],axis=1)


# In[611]:


# Checking the percentage of missing values
round(100*(LS_1.isnull().sum()/len(LS_1.index)), 2)


# In[612]:


LS_1.columns


# In[613]:


# Removing all the columns that are no required and have 35% null values
LS_2 = LS_1.drop(['Asymmetrique Profile Index','Asymmetrique Activity Index','Asymmetrique Activity Score','Asymmetrique Profile Score','Lead Profile','Tags','Lead Quality','How did you hear about X Education','City','Lead Number'],axis=1)
LS_2.head()


# In[614]:


# Rechecking the percentage of missing values
round(100*(LS_2.isnull().sum()/len(LS_2.index)), 2)


# ___There is a huge value of null variables in 4 columns as seen above. But removing the rows with the null value will cost us a lot of data and they are important columns. 
# So, instead we are going to replace the NaN values with 'not provided'. 
# This way we have all the data and almost no null values. 
# In case these come up in the model, it will be of no use and we can drop it off then.___

# In[615]:


LS_2['Specialization'] = LS_2['Specialization'].fillna('not provided') 
LS_2['What matters most to you in choosing a course'] = LS_2['What matters most to you in choosing a course'].fillna('not provided')
LS_2['Country'] = LS_2['Country'].fillna('not provided')
LS_2['What is your current occupation'] = LS_2['What is your current occupation'].fillna('not provided')
LS_2.info()


# In[616]:


# Rechecking the percentage of missing values
round(100*(LS_2.isnull().sum()/len(LS_2.index)), 2)


# In[617]:


LS_2["Country"].value_counts()


# In[618]:


def slots(x):
    category = ""
    if x == "india":
        category = "india"
    elif x == "not provided":
        category = "not provided"
    else:
        category = "outside india"
    return category

LS_2['Country'] = LS_2.apply(lambda x:slots(x['Country']), axis = 1)
LS_2['Country'].value_counts()


# In[619]:


# Rechecking the percentage of missing values
round(100*(LS_2.isnull().sum()/len(LS_2.index)), 2)


# In[620]:


# Checking the percent of lose if the null values are removed
round(100*(sum(LS_2.isnull().sum(axis=1) > 1)/LS_2.shape[0]),2)


# In[621]:


LS_3 = LS_2[LS_2.isnull().sum(axis=1) <1]

# Code for checking number of rows left in percent
round(100*(LS_3.shape[0])/(LS.shape[0]),2)


# In[622]:


# Rechecking the percentage of missing values
round(100*(LS_3.isnull().sum()/len(LS_3.index)), 2)


# In[623]:


# To familiarize all the categorical values
for column in LS_3:
    print(LS_3[column].astype('category').value_counts())
    print('----------------------------------------------------------------------------------------')


# In[624]:


# Removing Id values since they are unique for everyone
LS_final = LS_3.drop('Prospect ID',1)
LS_final.shape


# # B : EDA
# ## B.1 : Univariate Analysis
# ### B.1.1 : Categorical Variables

# In[625]:


LS_final.info()


# In[626]:


plt.figure(figsize = (20,40))

plt.subplot(4,3,1)
sns.countplot(LS_final['Lead Origin'])
plt.title('Lead Origin')

plt.subplot(4,3,2)
sns.countplot(LS_final['Do Not Email'])
plt.title('Do Not Email')

plt.subplot(4,3,3)
sns.countplot(LS_final['Do Not Call'])
plt.title('Do Not Call')

plt.subplot(4,3,4)
sns.countplot(LS_final['Country'])
plt.title('Country')

plt.subplot(4,3,5)
sns.countplot(LS_final['Search'])
plt.title('Search')

plt.subplot(4,3,6)
sns.countplot(LS_final['Newspaper Article'])
plt.title('Newspaper Article')

plt.subplot(4,3,7)
sns.countplot(LS_final['X Education Forums'])
plt.title('X Education Forums')

plt.subplot(4,3,8)
sns.countplot(LS_final['Newspaper'])
plt.title('Newspaper')

plt.subplot(4,3,9)
sns.countplot(LS_final['Digital Advertisement'])
plt.title('Digital Advertisement')

plt.subplot(4,3,10)
sns.countplot(LS_final['Through Recommendations'])
plt.title('Through Recommendations')

plt.subplot(4,3,11)
sns.countplot(LS_final['A free copy of Mastering The Interview'])
plt.title('A free copy of Mastering The Interview')

plt.subplot(4,3,12)
sns.countplot(LS_final['Last Notable Activity']).tick_params(axis='x', rotation = 90)
plt.title('Last Notable Activity')


plt.show()


# In[627]:


sns.countplot(LS_final['Lead Source']).tick_params(axis='x', rotation = 90)
plt.title('Lead Source')
plt.show()


# In[628]:


plt.figure(figsize = (20,30))
plt.subplot(2,2,1)
sns.countplot(LS_final['Specialization']).tick_params(axis='x', rotation = 90)
plt.title('Specialization')
plt.subplot(2,2,2)
sns.countplot(LS_final['What is your current occupation']).tick_params(axis='x', rotation = 90)
plt.title('Current Occupation')
plt.subplot(2,2,3)
sns.countplot(LS_final['What matters most to you in choosing a course']).tick_params(axis='x', rotation = 90)
plt.title('What matters most to you in choosing a course')
plt.subplot(2,2,4)
sns.countplot(LS_final['Last Activity']).tick_params(axis='x', rotation = 90)
plt.title('Last Activity')
plt.show()


# In[629]:


sns.countplot(LS['Converted'])
plt.title('Converted("Y variable")')
plt.show()


# ### B.1.2 : Numerical Variables

# In[630]:


LS_final.info()


# In[631]:


plt.figure(figsize = (20,10))
plt.subplot(221)
plt.hist(LS_final['TotalVisits'], bins = 200)
plt.title('Total Visits')
plt.xlim(0,25)

plt.subplot(222)
plt.hist(LS_final['Total Time Spent on Website'], bins = 10)
plt.title('Total Time Spent on Website')

plt.subplot(223)
plt.hist(LS_final['Page Views Per Visit'], bins = 20)
plt.title('Page Views Per Visit')
plt.xlim(0,20)
plt.show()


# ## B.2 : Relating all the categorical variables to Converted

# In[632]:


plt.figure(figsize = (20,10))

plt.subplot(1,2,1)
sns.countplot(x='Lead Origin', hue='Converted', data= LS_final).tick_params(axis='x', rotation = 90)
plt.title('Lead Origin')

plt.subplot(1,2,2)
sns.countplot(x='Lead Source', hue='Converted', data= LS_final).tick_params(axis='x', rotation = 90)
plt.title('Lead Source')
plt.show()


# In[633]:


plt.figure(figsize = (20,10))

plt.subplot(1,2,1)
sns.countplot(x='Do Not Email', hue='Converted', data= LS_final).tick_params(axis='x', rotation = 45)
plt.title('Do Not Email')

plt.subplot(1,2,2)
sns.countplot(x='Do Not Call', hue='Converted', data= LS_final).tick_params(axis='x', rotation = 45)
plt.title('Do Not Call')
plt.show()


# In[634]:


plt.figure(figsize = (20,10))

plt.subplot(1,2,1)
sns.countplot(x='Last Activity', hue='Converted', data= LS_final).tick_params(axis='x', rotation = 75)
plt.title('Last Activity')

plt.subplot(1,2,2)
sns.countplot(x='Country', hue='Converted', data= LS_final).tick_params(axis='x', rotation = 75)
plt.title('Country')
plt.show()


# In[635]:


plt.figure(figsize = (20,10))

plt.subplot(1,2,1)
sns.countplot(x='Specialization', hue='Converted', data= LS_final).tick_params(axis='x', rotation = 75)
plt.title('Specialization')

plt.subplot(1,2,2)
sns.countplot(x='What is your current occupation', hue='Converted', data= LS_final).tick_params(axis='x', rotation = 75)
plt.title('What is your current occupation')
plt.show()


# In[636]:


plt.figure(figsize = (20,10))

plt.subplot(1,2,1)
sns.countplot(x='What matters most to you in choosing a course', hue='Converted', data= LS_final).tick_params(axis='x', rotation = 45)
plt.title('What matters most to you in choosing a course')

plt.subplot(1,2,2)
sns.countplot(x='Search', hue='Converted', data= LS_final).tick_params(axis='x', rotation = 45)
plt.title('Search')
plt.show()


# In[637]:


plt.figure(figsize = (20,10))

plt.subplot(1,2,1)
sns.countplot(x='Newspaper Article', hue='Converted', data= LS_final).tick_params(axis='x', rotation = 45)
plt.title('Newspaper Article')

plt.subplot(1,2,2)
sns.countplot(x='X Education Forums', hue='Converted', data= LS_final).tick_params(axis='x', rotation = 45)
plt.title('X Education Forums')
plt.show()


# In[638]:


plt.figure(figsize = (20,10))

plt.subplot(1,2,1)
sns.countplot(x='Newspaper', hue='Converted', data= LS_final).tick_params(axis='x', rotation = 90)
plt.title('Newspaper')

plt.subplot(1,2,2)
sns.countplot(x='Digital Advertisement', hue='Converted', data= LS_final).tick_params(axis='x', rotation = 90)
plt.title('Digital Advertisement')
plt.show()


# In[639]:


plt.figure(figsize = (20,10))

plt.subplot(2,2,1)
sns.countplot(x='Through Recommendations', hue='Converted', data= LS_final).tick_params(axis='x', rotation = 45)
plt.title('Through Recommendations')

plt.subplot(2,2,2)
sns.countplot(x='A free copy of Mastering The Interview', hue='Converted', data= LS_final).tick_params(axis='x', rotation = 45)
plt.title('A free copy of Mastering The Interview')

plt.show()


# In[640]:


sns.countplot(x='Last Notable Activity', hue='Converted', data= LS_final).tick_params(axis='x', rotation = 75)
plt.title('Last Notable Activity')
plt.show()


# ##  The correlation among varibles

# In[641]:


# To check the correlation among varibles
plt.figure(figsize=(10,5))
sns.heatmap(LS_final.corr(),annot=True)
plt.show()


# *__It is understandable from the above EDA that there are many elements that have very little data and so will be of less relevance to our analysis.__*

# In[642]:


numeric = LS_final[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']]
numeric.describe(percentiles=[0.25,0.5,0.75,0.9,0.99])


# ___There aren't any major outliers, so moving on to analysis___

# # C : Dummy Variables

# In[643]:


LS_final.info()


# In[644]:


LS_final.loc[:, LS_final.dtypes == 'object'].columns


# In[645]:


# Create dummy variables using the 'get_dummies'
dummy = pd.get_dummies(LS_final[['Lead Origin','Specialization' ,'Lead Source', 'Do Not Email', 'Last Activity', 'What is your current occupation','A free copy of Mastering The Interview', 'Last Notable Activity']], drop_first=True)
# Add the results to the master dataframe
LS_dum = pd.concat([LS_final, dummy], axis=1)
LS_dum.head()


# In[646]:


LS_dum.shape


# In[647]:


LS_dum = LS_dum.drop(['What is your current occupation_not provided','Lead Origin', 'Lead Source', 'Do Not Email', 'Do Not Call','Last Activity', 'Country', 'Specialization', 'Specialization_not provided','What is your current occupation','What matters most to you in choosing a course', 'Search','Newspaper Article', 'X Education Forums', 'Newspaper','Digital Advertisement', 'Through Recommendations','A free copy of Mastering The Interview', 'Last Notable Activity'], 1)
LS_dum


# In[648]:


LS_dum.shape


# # D : Test-Train Split
# ## Import the required library

# In[649]:


from sklearn.model_selection import train_test_split

# Import MinMax scaler
from sklearn.preprocessing import MinMaxScaler


# In[650]:


X = LS_dum.drop(['Converted'], 1)
X.head()


# In[651]:


X.info()


# In[672]:


# Putting the target variable in y
y = LS_dum['Converted']
y.head()


# ### Split the dataset into 70% and 30% for train and test respectively

# In[673]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[ ]:





# In[674]:


# Scale the three numeric features
scaler = MinMaxScaler()
X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.fit_transform(X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])
X_train.head()


# In[675]:


# To check the correlation among varibles
plt.figure(figsize=(20,30))
sns.heatmap(X_train.corr())
plt.show()


# ___Since there are a lot of variables it is difficult to drop variable. We'll do it after RFE___

# # E :  Model Building

# ## Import 'LogisticRegression'
# ## Import RFE

# In[676]:


# Import 'LogisticRegression'
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[677]:


# Import RFE
from sklearn.feature_selection import RFE


# In[681]:


# Running RFE with 15 variables as output
rfe = RFE( estimator=LogisticRegression(), n_features_to_select = 15)
rfe = rfe.fit(X_train, y_train)


# In[682]:


rfe.support_


# In[683]:


# Features that have been selected by RFE
list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[684]:


# Put all the columns selected by RFE in the variable 'col'
col = X_train.columns[rfe.support_]


# ___All the variables selected by RFE, next statistics part (p-values and the VIFs).___

# In[685]:


# Selecting columns selected by RFE
X_train = X_train[col]


# # Importing statsmodels
# 

# In[686]:


import statsmodels.api as sm


# In[687]:


X_train_sm = sm.add_constant(X_train)
logm1 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# ### Importing 'variance_inflation_factor'

# In[688]:


# Importing 'variance_inflation_factor'
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[689]:


# Make a VIF dataframe for all the variables present
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ___The VIF values seem fine but the p-values aren't. So removing 'What is your current occupation_housewife'___

# In[690]:


X_train.drop('What is your current occupation_housewife', axis = 1, inplace = True)


# In[691]:


# Refit the model with the new set of features
X_train_sm = sm.add_constant(X_train)
logm2 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[692]:


# Make a VIF dataframe for all the variables present
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ___The VIF values seem fine but the p-values aren't. So removing 'Lead Source_referral sites'___

# In[693]:


X_train.drop('Lead Source_referral sites', axis = 1, inplace = True)


# In[694]:


# Refit the model with the new set of features
X_train_sm = sm.add_constant(X_train)
logm2 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[695]:


# Make a VIF dataframe for all the variables present
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ___All the VIF values are good and all the p-values are below 0.05. So we can fix model.___

# # F :  Creating Prediction

# In[696]:


# Predicting the probabilities on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[697]:


# Reshaping to an array
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[698]:


# Data frame with given convertion rate and probablity of predicted ones
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Conversion_Prob':y_train_pred})
y_train_pred_final.head()


# In[699]:


# Substituting 0 or 1 with the cut off as 0.5
y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# # G : Model Evaluation
# ### Importing metrics from sklearn for evaluation

# In[700]:


from sklearn import metrics


# In[701]:


# Creating confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
confusion


# In[705]:


# Predicted   |  not_churn  |  churn
#-------------|             |
# Actual      |             |
#-----------------------------------------
# not_churn   |     3403    |   492
#------------------------------------------
# churn       |      729    |  1727


# In[706]:


# Check the overall accuracy
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted)


# ___That's around 80% accuracy with is a very good value___

# In[707]:


# Substituting the value of true positive
TP = confusion[1,1]
# Substituting the value of true negatives
TN = confusion[0,0]
# Substituting the value of false positives
FP = confusion[0,1] 
# Substituting the value of false negatives
FN = confusion[1,0]


# In[708]:


# Calculating the sensitivity
TP/(TP+FN)


# In[709]:


# Calculating the specificity
TN/(TN+FP)


# ___With the current cut off as 0.5 we have around 80% accuracy, sensitivity of around 66% and specificity of around 88%.___

# # H : Optimise Cut off (ROC Curve)
# ### The previous cut off was randomely selected. Now to find the optimum one

# In[710]:


# ROC function
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[711]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob, drop_intermediate = False )


# In[712]:


# Call the ROC function
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# ___The area under ROC curve is 0.87 which is a very good value.___

# In[714]:


# Creating columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[716]:


# Creating a dataframe to see the values of accuracy, sensitivity, and specificity at different values of probabiity cutoffs
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
# Making confusing matrix to find values of sensitivity, accurace and specificity for each level of probablity
from sklearn.metrics import confusion_matrix
num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
cutoff_df


# In[717]:


# Plotting it
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# ___From the graph it is visible that the optimal cut off is at 0.38.___

# In[724]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map( lambda x: 1 if x > 0.38 else 0)
y_train_pred_final.head()


# In[725]:


# Check the overall accuracy
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[726]:


# Creating confusion matrix 
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[727]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[728]:


# Calculating the sensitivity
TP/(TP+FN)


# In[729]:


# Calculating the specificity
TN/(TN+FP)


# ___With the current cut off as 0.38 we have accuracy, sensitivity and specificity of around 80%.___

# # I : Prediction on Test set

# In[732]:


# Scaling numeric values
X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.transform(X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])


# In[733]:


# Substituting all the columns in the final train model
col = X_train.columns


# In[734]:


# Select the columns in X_train for X_test as well
X_test = X_test[col]
# Add a constant to X_test
X_test_sm = sm.add_constant(X_test[col])
X_test_sm
X_test_sm


# In[735]:


# Storing prediction of test set in the variable 'y_test_pred'
y_test_pred = res.predict(X_test_sm)
# Coverting it to df
y_pred_df = pd.DataFrame(y_test_pred)
# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
# Remove index for both dataframes to append them side by side 
y_pred_df.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Append y_test_df and y_pred_df
y_pred_final = pd.concat([y_test_df, y_pred_df],axis=1)
# Renaming column 
y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})
y_pred_final.head()


# In[748]:


# Making prediction using cut off 0.38
y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.38 else 0)
y_pred_final


# In[749]:


# Check the overall accuracy
metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[750]:


# Creating confusion matrix 
confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[780]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[781]:


# Calculating the sensitivity
TP/(TP+FN)


# In[782]:


# Calculating the specificity
TN/(TN+FP)


# ___With the current cut off as 0.38 we have accuracy, sensitivity around 50% and specificity around 93%.___

# # J : Precision-Recall

# In[754]:


confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
confusion


# In[756]:


# Precision = TP / TP + FP
confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[757]:


#Recall = TP / TP + FN
confusion[1,1]/(confusion[1,0]+confusion[1,1])


# ___With the current cut off as 0.38 we have Precision around 77% and Recall around 66%___

# # J.1 : Precision and recall tradeoff 
# 
# ### Import precision_recall_curve

# In[758]:


from sklearn.metrics import precision_recall_curve


# In[759]:


y_train_pred_final.Converted, y_train_pred_final.Predicted


# In[760]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# In[763]:


plt.plot(thresholds, p[:-1], "r-")
plt.plot(thresholds, r[:-1], "b-")
plt.show()


# In[765]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.42 else 0)
y_train_pred_final.head()


# In[766]:


# Accuracy
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[767]:


# Creating confusion matrix again
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[768]:


# Substituting the value of true positive
TP = confusion2[1,1]
# Substituting the value of true negatives
TN = confusion2[0,0]
# Substituting the value of false positives
FP = confusion2[0,1] 
# Substituting the value of false negatives
FN = confusion2[1,0]


# In[769]:


# Precision = TP / TP + FP
TP / (TP + FP)


# In[770]:


#Recall = TP / TP + FN
TP / (TP + FN)


# ___With the current cut off as 0.42 we have Precision around 74% and Recall around 78%___

# # K : Prediction on Test set

# In[771]:


# Storing prediction of test set in the variable 'y_test_pred'
y_test_pred = res.predict(X_test_sm)
# Coverting it to df
y_pred_df = pd.DataFrame(y_test_pred)
# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
# Remove index for both dataframes to append them side by side 
y_pred_df.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Append y_test_df and y_pred_df
y_pred_final = pd.concat([y_test_df, y_pred_df],axis=1)
# Renaming column 
y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})
y_pred_final.head()


# In[772]:


# Making prediction using cut off 0.41
y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.42 else 0)
y_pred_final


# In[773]:


# Check the overall accuracy
metrics.accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted)


# In[774]:


# Creating confusion matrix 
confusion2 = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted )
confusion2


# In[775]:


# Precision = TP / TP + FP
TP / (TP + FP)


# In[776]:


#Recall = TP / TP + FN
TP / (TP + FN)


# ___With the current cut off as 0.42 we have Precision around 74% and Recall around 78%___

# # L : 
# # Conclusion :-
# ### It was found that the variables that mattered the most in the potential buyers are__ (In descending order) :
# 
# * 1. The total time spend on the Website.
# 
# * 2. When the Lead Origin_lead add form.
# 
# * 3. When the lead source was:
#  *   a. Direct traffic,
#  *   b. Welingak website.
#  
# * 4. When the click "Yes!" at Do Not Email option.
#  
# * 5. When the Last Activity was:
#  *   a. had a phone conversation,
#  *   b. olark chat conversation.
# 
# * 6. When their current occupation is as a working professional.
# 
# * 7. When the Last Notable Activity was :
#  *   a. email link clicked,
#  *   b. email opened,
#  *   c. modified,
#  *   d. olark chat conversation,
#  *   e. page visited on website.
# 
# ### Keeping these in mind the X Education can flourish as they have a very high chance to get almost all the potential buyers to change their mind and buy their courses.

# In[ ]:




