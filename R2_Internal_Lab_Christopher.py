#!/usr/bin/env python
# coding: utf-8

# # K nearest neighbors

# KNN falls in the supervised learning family of algorithms. Informally, this means that we are given a labelled dataset consiting of training observations (x, y) and would like to capture the relationship between x and y. More formally, our goal is to learn a function h: X→Y so that given an unseen observation x, h(x) can confidently predict the corresponding output y.
# 
# In this module we will explore the inner workings of KNN, choosing the optimal K values and using KNN from scikit-learn.

# ## Overview
# 
# 1. Read the problem statement.
# 
# 2. Get the dataset.
# 
# 3. Explore the dataset.
# 
# 4. Pre-processing of dataset.
# 
# 5. Visualization
# 
# 6. Transform the dataset for building machine learning model.
# 
# 7. Split data into train, test set.
# 
# 8. Build Model.
# 
# 9. Apply the model.
# 
# 10. Evaluate the model.
# 
# 11. Finding Optimal K value
# 
# 12. Repeat 7, 8, 9 steps.

# ### Dataset
# 
# The data set we’ll be using is the Iris Flower Dataset which was first introduced in 1936 by the famous statistician Ronald Fisher and consists of 50 observations from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals.
# 
# **Download the dataset here:**
# - https://www.kaggle.com/uciml/iris
# 
# **Train the KNN algorithm to be able to distinguish the species from one another given the measurements of the 4 features.**

# ## Load data

# ### Question 1
# 
# Import the data set and print 10 random rows from the data set
# 
# Hint: use **sample()** function to get random rows

# In[ ]:





# ## Data Pre-processing

# ### Question 2 - Estimating missing values
# 
# Its not good to remove the records having missing values all the time. We may end up loosing some data points. So, we will have to see how to replace those missing values with some estimated values (median)

# Calculate the number of missing values per column
# - don't use loops

# In[ ]:





# Fill missing values with median of that particular column

# In[ ]:





# ### Question 3 - Dealing with categorical data
# 
# Change all the classes to numericals (0 to 2)
# 
# Hint: use **LabelEncoder()**

# In[ ]:





# ### Question 4
# 
# Observe the association of each independent variable with target variable and drop variables from feature set having correlation in range -0.1 to 0.1 with target variable.
# 
# Hint: use **corr()**

# In[ ]:





# ### Question 5
# 
# Observe the independent variables variance and drop such variables having no variance or almost zero variance (variance < 0.1). They will be having almost no influence on the classification
# 
# Hint: use **var()**

# In[ ]:





# ### Question 6
# 
# Plot the scatter matrix for all the variables.
# 
# Hint: use **pandas.plotting.scatter_matrix()**
# 
# you can also use pairplot()

# In[ ]:





# ## Split the dataset into training and test sets
# 

# ### Question 7
# 
# Split the dataset into training and test sets with 80-20 ratio
# 
# Hint: use **train_test_split()**

# In[ ]:





# ## Build Model

# ### Question 8
# 
# Build the model and train and test on training and test sets respectively using **scikit-learn**.
# 
# Print the Accuracy of the model with different values of **k = 3, 5, 9**
# 
# Hint: For accuracy you can check **accuracy_score()** in scikit-learn

# In[ ]:





# ## Find optimal value of K

# ### Question 9 - Finding Optimal value of k
# 
# - Run the KNN with no of neighbours to be 1, 3, 5 ... 19
# - Find the **optimal number of neighbours** from the above list

# In[ ]:





# ## Plot accuracy

# ### Question 10
# 
# Plot accuracy score vs k (with k value on X-axis) using matplotlib.

# In[ ]:





# In[ ]:





# # Breast cancer dataset

# ## Read data

# ### Question 1
# Read the data given in bc2.csv file

# In[209]:


import pandas as pd
import numpy as np
bc = pd.read_csv("bc2.csv")
bc.head()


# ## Data preprocessing

# ### Question 2
# Observe the no.of records in dataset and type of each column

# In[142]:


bc.shape


# In[143]:


bc.dtypes


# ### Question 3
# Use summary statistics to check if missing values, outlier and encoding treament is necessary
# 
# Hint: use **describe()**

# In[144]:


bc.describe()


# #### Check Missing Values

# In[145]:


bc.isna().sum()


# ### Question 4
# #### Check how many `?` are there in Bare Nuclei feature (they are also unknown or missing values). 

# In[146]:


len(bc[bc['Bare Nuclei']=='?'])


# #### Replace them with the 'top' value of the describe function of Bare Nuclei feature
# 
# Hint: give value of parameter include='all' in describe function

# In[210]:


Bnt=bc.describe(include="all")
bc[bc['Bare Nuclei']=='?']
Bnt['Bare Nuclei'].top
Bnt['Bare Nuclei'].shape[0]
bc.replace("?",Bnt['Bare Nuclei'].top,inplace=True)


# ### Question 5
# #### Find the distribution of target variable (Class) 

# In[148]:


import matplotlib.pyplot as plt
bc['Class'].value_counts().plot.bar()
plt.show()


# #### Plot the distribution of target variable using histogram

# In[149]:


bc['Class'].hist()
plt.title("Class")
plt.show()


# #### Convert the datatype of Bare Nuclei to `int`

# In[211]:


bc['Bare Nuclei']=bc['Bare Nuclei'].astype(int)
bc['Bare Nuclei'].dtypes


# ## Scatter plot

# ### Question 6
# Plot Scatter Matrix to understand the distribution of variables and check if any variables are collinear and drop one of them.

# In[151]:


from matplotlib import pyplot as plt
#!pip install seaborn
import seaborn as sns
sns.set(style="ticks",color_codes="True")
sns.pairplot(bc)


# In[212]:


bc.corr()
bc.drop(columns=['Cell Shape'],inplace=True)
bc


# ## Train test split

# # Question 7
# #### Divide the dataset into feature set and target set

# In[213]:


X=bc.drop(columns="Class")
y=bc["Class"]
X.dtypes


# #### Divide the Training and Test sets in 70:30 

# In[214]:


import warnings
warnings.filterwarnings("ignore")


# In[215]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=1)


# ## Scale the data

# ### Question 8
# Standardize the data
# 
# Hint: use **StandardScaler()**

# In[216]:


sc=StandardScaler()
sc.fit(X_train)
scaledX_train = sc.transform(X_train)
scaledX_test = sc.transform(X_test)


# ## Build Model

# ### Question 9
# 
# Build the model and train and test on training and test sets respectively using **scikit-learn**.
# 
# Print the Accuracy of the model with different values of **k = 3, 5, 9**
# 
# Hint: For accuracy you can check **accuracy_score()** in scikit-learn

# In[217]:


knn.fit(scaledX_train,y_train)
print("What is the Testing Accuracy")
print(knn.score(scaledX_test,y_test))
print("What is the Training Accuracy")
print(knn.score(scaledX_train,y_train))


# ## Find optimal value of K

# ### Question 10
# Finding Optimal value of k
# 
# - Run the KNN with no of neighbours to be 1, 3, 5 ... 19
# - Find the **optimal number of neighbours** from the above list

# In[223]:


neighbors = np.arange(1,19,2)
train_accuracy_plot = np.empty(len(neighbors))
test_accuracy_plot = np.empty(len(neighbors))
# Loop over different values of k
for i, k in enumerate(neighbors):
    train = []
    test = []
    for j in range(20):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=j)
        sc=StandardScaler()
        scaledX_train = sc.fit_transform(X_train)
        scaledX_test = sc.transform(X_test)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(scaledX_train,y_train)
        train.append(knn.score(scaledX_train,y_train))
        test.append(knn.score(scaledX_test,y_test))
    #Compute accuracy on the training set
    train_accuracy_plot[i] = np.mean(train)
    #Compute accuracy on the testing set
    test_accuracy_plot[i] = np.mean(test)


# ## Plot accuracy

# ### Question 11
# 
# Plot accuracy score vs k (with k value on X-axis) using matplotlib.

# In[225]:


## plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy_plot, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy_plot, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

