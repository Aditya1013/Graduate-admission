
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt     #data visualization
import seaborn as sns


# In[16]:


from sklearn.model_selection import train_test_split


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


os.chdir('C:/Users/Adi/Desktop/kaggle/graduate-admissions')


# In[4]:


admission_in_universities = pd.read_csv('Admission_Predict.csv')


# In[5]:


admission_in_universities.head()


# In[6]:


admission_in_universities.drop('Serial No.', axis = 1, inplace = True)


# In[7]:


admission_in_universities.head()


# In[8]:


admission_in_universities.describe()


# In[9]:


plt.figure(figsize = (10,6))
sns.heatmap(admission_in_universities.corr(), annot = True, linewidths=0.5, cmap = 'coolwarm')
plt.show()


# In[10]:


sns.set_style('darkgrid')
sns.pairplot(admission_in_universities)
plt.show()


# In[11]:


sns.countplot(admission_in_universities['University Rating'])
plt.show()


# In[12]:


sns.jointplot(x = 'CGPA', y ='Chance of Admit ', data = admission_in_universities)
plt.show()


# In[13]:


plt.figure(figsize=(10,6))
sns.distplot(admission_in_universities['TOEFL Score'], kde = False, bins = 30, color = 'blue')
plt.show()


# In[14]:


plt.figure(figsize=(10,6))
sns.distplot(admission_in_universities['GRE Score'], kde = False, bins = 30, color = 'red')
plt.show()


# In[18]:


X = admission_in_universities[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',
       'Research']]
y = admission_in_universities['Chance of Admit ']


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.35, random_state = 101)


# In[19]:


from sklearn.linear_model import LinearRegression


# In[20]:


linearmodel = LinearRegression()


# In[21]:


linearmodel.fit(X_train,y_train)


# In[22]:


cadmission_in_universities = pd.DataFrame(linearmodel.coef_, X.columns, columns=['Coefficient'])
cadmission_in_universities


# In[23]:


prediction = linearmodel.predict(X_test)


# In[24]:


plt.figure(figsize=(9,6))
plt.scatter(y_test,prediction)
plt.xlabel('Test Values for y')
plt.ylabel('Predicted Values for y')
plt.title('Scatter Plot of Real Test Values vs. Predicted Values ')
plt.show()


# In[25]:


from sklearn import metrics
print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test,prediction))
print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, prediction))
print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test,prediction)))

