#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


# In[5]:


file_path = r'C:\Users\rites\OneDrive\Desktop\Anonymize_Loan_Default_data.csv'
df = pd.read_csv(file_path, encoding='latin1')


# In[6]:


df.head()


# In[7]:


def give_stats(data):
    return data.describe()

def give_missing(data):
    return data.isnull().sum()

def give_datatype(data):
    return data.dtypes

print("Summary Stats:")
print(give_stats(df))

print("\nMissing Values:")
print(give_missing(df))

print("\nColumns Data Type:")
print(give_datatype(df))


# In[8]:


int_columns = []
float_columns = []
str_columns = []
missing_columns = []

for col in df.columns:
    dtype = df[col].dtype
    
    if dtype == 'int64':
        int_columns.append(col)
        
for col in df.columns:
    dtype = df[col].dtype
    
    if dtype == 'float64':
        float_columns.append(col)
    
for col in df.columns:
    dtype = df[col].dtype
    
    if dtype == 'object':
        str_columns.append(col)

def missing_value_column(column):
    return df[column].isnull().sum()

for col in df.columns:
    missing_count = missing_value_column(col)
    if missing_count > 0:
        missing_columns.append((col, missing_count))

result = {
    'Integer Columns': int_columns,
    'Float Columns': float_columns,
    'String Columns': str_columns,
    'Columns with Missing Values': missing_columns
}

print(result)

def plot_missing_val_column(missing_columns):
    missing_columns_df = pd.DataFrame(missing_columns, columns=['Column', 'Missing Values'])
    missing_columns_df.set_index('Column', inplace=True)

    plt.figure(figsize=(10, 6))
    plt.bar(missing_columns_df.index, missing_columns_df['Missing Values'], color='coral')
    plt.title('Missing Values in Columns')
    plt.xlabel('Column Name')
    plt.ylabel('Number of Missing Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_missing_val_column(missing_columns)


# In[9]:


df = df.drop(df.columns[0], axis=1)


# In[10]:


df.head()


# In[11]:


df=df.drop(df.columns[33],axis=1)
df=df.drop(df.columns[21],axis=1)
df.head()


# In[12]:


def remove_missing(data):
    return df.dropna()

df = remove_missing(df)


# In[13]:


def process_and_convert(value):
    numeric_value = ''.join(filter(str.isdigit, value))
    return int(numeric_value)/100

df['revol_util'] = df['revol_util'].apply(process_and_convert)

print(df)


# In[14]:


df['earliest_cr_line']=pd.to_datetime(df['earliest_cr_line'],format='%b-%y')
df['earliest_cr_line_M'] = df['earliest_cr_line'].dt.month
df['earliest_cr_line_Y'] = df['earliest_cr_line'].dt.year
df['issue_d']=pd.to_datetime(df['issue_d'],format='%b-%y')
df['issue_d_M'] = df['issue_d'].dt.month
df['issue_d_Y'] = df['issue_d'].dt.year
df['last_pymnt_d']=pd.to_datetime(df['last_pymnt_d'],format='%b-%y')
df['last_pymnt_d_M'] = df['last_pymnt_d'].dt.month
df['last_pymnt_d_Y'] = df['last_pymnt_d'].dt.year
df['last_credit_pull_d']=pd.to_datetime(df['last_credit_pull_d'],format='%b-%y')
df['last_credit_pull_d_M'] = df['last_credit_pull_d'].dt.month
df['last_credit_pull_d_Y'] = df['last_credit_pull_d'].dt.year

df = df.drop(columns=["earliest_cr_line","issue_d","last_pymnt_d","last_credit_pull_d"],axis=1)
print(df.info())


# In[15]:


df['term'] = df['term'].str.extract('(\d+)').astype(int)

emp_length_mapping = {'0 year': 0, '< 1 year': 0.5,'1 year': 1,'2 years': 2,'3 years': 3,'4 years': 4,
    '5 years': 5,'6 years': 6,'7 years': 7,'8 years': 8,'9 years': 9,'10 years': 10,'10+ years': 11 }
df['emp_length'] = df['emp_length'].map(emp_length_mapping)
df['emp_length'] = df['emp_length'].astype(float)
print(df.info())


# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns

def num_column(data):
    d = df.select_dtypes(include=['number'])
    return d.columns

numerical_columns = num_column(df)
num_bins = 20

for column in numerical_columns:
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(df[column], bins=num_bins)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column}')
    plt.tight_layout()
    plt.show()
    


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns

num_bins = 20

num_columns_group1 = numerical_columns[:len(numerical_columns) // 4]
num_columns_group2 = numerical_columns[len(numerical_columns) // 4:2 * len(numerical_columns) // 4]
num_columns_group3 = numerical_columns[2 * len(numerical_columns) // 4:3 * len(numerical_columns) // 4]
num_columns_group4 = numerical_columns[3 * len(numerical_columns) // 4:]

plt.figure(figsize=(20, 8))

def make_boxplot(data,column):
    sns.boxplot(x='repay_fail', y=column, data=df)
    plt.xlabel('Target')
    plt.ylabel(column)
    plt.title(f'{column}')
    
for i, column in enumerate(num_columns_group1):
    plt.subplot(1, len(num_columns_group1), i + 1)
    
    make_boxplot(df,column)

plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 8))

for i, column in enumerate(num_columns_group2):
    plt.subplot(1, len(num_columns_group2), i + 1)
    
    make_boxplot(df,column)

plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 8))

for i, column in enumerate(num_columns_group3):
    plt.subplot(1, len(num_columns_group3), i + 1)
    
    make_boxplot(df,column)
    
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 8))

for i, column in enumerate(num_columns_group4):
    plt.subplot(1, len(num_columns_group4), i + 1)
    
    make_boxplot(df,column)
    
plt.tight_layout()
plt.show()


# In[18]:


categorical_columns = df.select_dtypes(include=['object']).columns

num_cols = len(categorical_columns)
num_rows = (num_cols + 1) // 2  

plt.figure(figsize=(15, 5 * num_rows))

def make_bar(column,data):
    sns.countplot(x=column, data=data)
    plt.xlabel(column)
    plt.ylabel('Count')  
    plt.title(f'Bar Plot of {column}')
    
for i, column in enumerate(categorical_columns, 1):
    plt.subplot(num_rows, 2, i)
    
    make_bar(column,df)
    
    plt.xticks(rotation=90)
    
    
plt.tight_layout()
plt.show()


# In[19]:


exclude = ['repay_fail']

def find_iqr(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    return Q3 - Q1, Q1,Q3
    
def find_limits(Q1, Q3, iqr):
    lower = Q1 - 1.5 * iqr
    upper = Q3 + 1.5 * iqr
    return lower, upper

def remove_numerical_outliers_iqr(df, column_name):
    IQR,q1,q3 = find_iqr(column_name)
    lower_limit, upper_limit =find_limits(q1, q3, IQR)
    
    df_filtered = df[(df[column_name] >= lower_limit) & (df[column_name] <= upper_limit)]
    return df_filtered


for column in numerical_columns:
    if column not in exclude:
        df = remove_numerical_outliers_iqr(df, column)
        
print("Modified DataFrame (Numerical Columns):")
print(df)

min_category_occurrences = 10

def remove_infrequent_categories(df, threshold):
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    def find_infrequent_category(col):
        category_counts = df[col].value_counts()
        infrequent_categories = category_counts[category_counts < threshold].index
        return infrequent_categories
    
    for column in categorical_columns:
        infrequent_categories = find_infrequent_category(column)
        df = df[~df[column].isin(infrequent_categories)]

    return df

df = remove_infrequent_categories(df, min_category_occurrences)

print(df)


# In[20]:


import pandas as pd

def remove_outliers_zscore_numeric(df, columns, threshold=3):
    cleaned_df = df.copy()
    outlier_indices = []
    def find_zscore(col):
        return (df[col] - df[col].mean()) / df[col].std()
            
    def outliers(df):
        return z_scores.abs() > threshold
        
    for column in columns:
        if df[column].dtype in ['int64', 'float64']:
            z_scores = find_zscore(column)
            
        outlier_indices.extend(df.index[outliers(df)])
        outlier = outliers(df)
        cleaned_df.loc[~outlier] 
            
    return cleaned_df, outlier_indices

cleaned_df_numeric, outlier_indices_numeric = remove_outliers_zscore_numeric(df, numerical_columns)

cleaned_df_categorical = remove_infrequent_categories(df, min_category_occurrences)


df = pd.concat([cleaned_df_numeric, cleaned_df_categorical])

print(df)


# In[21]:


import matplotlib.pyplot as plt
import seaborn as sns

num_bins = 20

num_columns_group1 = numerical_columns[:len(numerical_columns) // 4]
num_columns_group2 = numerical_columns[len(numerical_columns) // 4:2 * len(numerical_columns) // 4]
num_columns_group3 = numerical_columns[2 * len(numerical_columns) // 4:3 * len(numerical_columns) // 4]
num_columns_group4 = numerical_columns[3 * len(numerical_columns) // 4:]

plt.figure(figsize=(20, 8))

def make_boxplot(data,column):
    sns.boxplot(x='repay_fail', y=column, data=df)
    plt.xlabel('Target')
    plt.ylabel(column)
    plt.title(f'{column}')
    
for i, column in enumerate(num_columns_group1):
    plt.subplot(1, len(num_columns_group1), i + 1)
    
    make_boxplot(df,column)

plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 8))

for i, column in enumerate(num_columns_group2):
    plt.subplot(1, len(num_columns_group2), i + 1)
    
    make_boxplot(df,column)

plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 8))

for i, column in enumerate(num_columns_group3):
    plt.subplot(1, len(num_columns_group3), i + 1)
    
    make_boxplot(df,column)
    
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 8))

for i, column in enumerate(num_columns_group4):
    plt.subplot(1, len(num_columns_group4), i + 1)
    
    make_boxplot(df,column)
    
plt.tight_layout()
plt.show()


# In[22]:


def find_value(s):
    return df[s].value_counts()

print("Value Counts for target\n", find_value('repay_fail'))


# In[23]:


corr_matrix = df.corr()
def make_heatmap(matrix):
    return sns.heatmap(matrix, annot=True, cmap='coolwarm', linewidths=0.5)

plt.figure(figsize=(29, 8))
make_heatmap(corr_matrix)
plt.title('Correlation Heatmap')
plt.show()


# In[24]:


X = df.drop('repay_fail',axis=1)
y = df['repay_fail']


# In[25]:


correlations_with_target = X.corrwith(y)

def plot_bar(cwt):
    sns.barplot(x=cwt.index, y=cwt.values, palette='coolwarm')
    plt.title('Correlation of Variables with Target Variable')
    plt.xlabel('Variables')
    plt.ylabel('Correlation')

plt.figure(figsize=(10, 6))
plot_bar(correlations_with_target)
plt.xticks(rotation=90)  
plt.show()


# In[26]:


# Data types of columns
data_types = df.dtypes
#print datatype of each variable
print(data_types)


# In[27]:


X = X.drop('id',axis=1)
X = X.drop('member_id',axis=1)
X = X.drop('emp_length',axis=1)
X = X.drop('earliest_cr_line_Y',axis=1)
X = X.drop('issue_d_M',axis=1)
X = X.drop('last_pymnt_d_M',axis=1)
X = X.drop('last_credit_pull_d_M',axis=1)
print(X,y)


# In[28]:


X = X.drop('loan_amnt',axis=1)
X = X.drop('funded_amnt_inv',axis=1)
X = X.drop('installment',axis=1)
X = X.drop('total_pymnt',axis=1)
X = X.drop('total_pymnt_inv',axis=1) 


# In[29]:


# Data types of columns
data_types = X.dtypes
print(data_types)


# In[30]:


X1 = X.select_dtypes(include=['number'])
numerical_columns = X1.columns
scaler = StandardScaler()

def standrize(f,col):
    f[col] = scaler.fit_transform(f[col])
    
standrize(X,numerical_columns)


# In[31]:


print(y.info())


# In[32]:


X2 = X.select_dtypes(include=['object'])

categorical_columns = X2.columns


X_categorical = pd.get_dummies(X, columns=categorical_columns)


X = pd.concat([X.drop(categorical_columns, axis=1), X_categorical], axis=1)

print(X,y)


# In[33]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[34]:


pip install pca


# In[35]:


from sklearn.decomposition import PCA

pca = PCA(n_components=100)

def perform_pca(pca,train,test):
    train_pca = pca.fit_transform(train)
    test_pca = pca.transform(test)
    return train_pca, test_pca

X_train_pca,X_test_pca = perform_pca(pca,X_train, X_test)


# In[36]:


print(y)


# In[37]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


# In[38]:


from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  
}

knn_model = KNeighborsClassifier()

grid_search = GridSearchCV(estimator=knn_model, param_grid=param_grid, cv=5, scoring= 'accuracy')
grid_search.fit(X_train_pca, y_train)

print("Best hyperparameters:", grid_search.best_params_)

weights_mapping = {'uniform': 0, 'distance': 1}
numerical_weights = [weights_mapping[w] for w in grid_search.cv_results_['param_weights']]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

x_vals = grid_search.cv_results_['param_n_neighbors']
y_vals = numerical_weights
z_vals = grid_search.cv_results_['param_p']

scatter = ax.scatter3D(x_vals, y_vals, z_vals, c=grid_search.cv_results_['mean_test_score'], cmap='viridis', marker='o')
ax.set_xlabel('n_neighbors')
ax.set_ylabel('weights')
ax.set_zlabel('p')
ax.set_title('Grid Search Results for KNN')

ax.set_yticks([0, 1])
ax.set_yticklabels(['uniform', 'distance'])

cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')
cbar.set_label('Mean Test Score', rotation=270, labelpad=15)

plt.show()


# In[39]:


param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 100],
    'solver': ['liblinear', 'saga']
}

logreg_model = LogisticRegression()

grid_search = GridSearchCV(estimator=logreg_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train_pca, y_train)

print("Best Parameters:\n", grid_search.best_params_)

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring= 'accuracy', n_jobs=-1)
grid_search.fit(X_train_pca, y_train)

penalty_mapping = {'l1': 0, 'l2': 1}
solver_mapping = {'liblinear': 0, 'saga': 1}

numerical_penalty = [penalty_mapping[p] for p in grid_search.cv_results_['param_penalty']]
numerical_solver = [solver_mapping[s] for s in grid_search.cv_results_['param_solver']]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

x_vals = numerical_penalty
y_vals = grid_search.cv_results_['param_C']
z_vals = numerical_solver

scatter = ax.scatter3D(x_vals, y_vals, z_vals, c=grid_search.cv_results_['mean_test_score'], cmap='viridis', marker='o')
ax.set_xlabel('penalty')
ax.set_ylabel('C')
ax.set_zlabel('solver')
ax.set_title('Grid Search Results for Logistic Regression')

ax.set_xticks([0, 1])
ax.set_xticklabels(['l1', 'l2'])
ax.set_zticks([0, 1])
ax.set_zticklabels(['liblinear', 'saga'])

cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')
cbar.set_label('Mean Test Score', rotation=270, labelpad=15)

plt.show()


# In[ ]:


import itertools
param_grid = {
    'n_estimators': [50, 100, 200],  
    'max_depth': [10, None],  
    'min_samples_split': [2, 5, 10]  
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, n_jobs=-1, scoring= 'accuracy')

grid_search.fit(X_train_pca, y_train)

print("Best Parameters:\n", grid_search.best_params_)

param_combinations = list(itertools.product(param_grid['n_estimators'], param_grid['max_depth'], param_grid['min_samples_split']))
mean_test_scores = grid_search.cv_results_['mean_test_score']

# Plotting the accuracy for each parameter combination
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

x_vals, y_vals, z_vals = zip(*param_combinations)

scatter = ax.scatter3D(x_vals, y_vals, z_vals, c=mean_test_scores, cmap='viridis', marker='o')
ax.set_xlabel('n_estimators')
ax.set_ylabel('max_depth')
ax.set_zlabel('min_samples_split')
ax.set_title('Grid Search Results for Random Forest Classifier')

cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')
cbar.set_label('Mean Test Score', rotation=270, labelpad=15)

plt.show()


# In[ ]:


param_grid = {
    'max_depth': [3, 5, None],  
    'min_samples_split': [2, 6],  
    'min_samples_leaf': [1, 2, 3]  
}

grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, n_jobs=-1, scoring= 'accuracy')

grid_search.fit(X_train_pca, y_train)

print("Best Parameters:\n", grid_search.best_params_)

param_combinations = list(itertools.product(param_grid['max_depth'], param_grid['min_samples_split'], param_grid['min_samples_leaf']))
mean_test_scores = grid_search.cv_results_['mean_test_score']

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

x_vals, y_vals, z_vals = zip(*param_combinations)

scatter = ax.scatter3D(x_vals, y_vals, z_vals, c=mean_test_scores, cmap='viridis', marker='o')
ax.set_xlabel('max_depth')
ax.set_ylabel('min_samples_split')
ax.set_zlabel('min_samples_leaf')
ax.set_title('Grid Search Results for Decision Tree')

cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')
cbar.set_label('Mean Test Score', rotation=270, labelpad=15)

plt.show()


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, auc
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

models = {
    'Logistic Regression': LogisticRegression(C= 100, penalty= 'l1', solver= 'liblinear'),
    'Random Forest': RandomForestClassifier(max_depth= None, min_samples_split= 5, n_estimators= 100),
    'Decision Tree': DecisionTreeClassifier(max_depth= 3, min_samples_leaf= 2, min_samples_split= 2),
    'KNN': KNeighborsClassifier(n_neighbors= 7, p= 2, weights ='uniform'),
}

plt.figure(figsize=(8, 6))

for model_name, model in models.items():
    model.fit(X_train_pca, y_train)

    y_pred = model.predict(X_test_pca)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Confusion Matrix:\n{confusion}\n")
    print(f"AUC-PR: {auc_pr:.2f}")


for model_name, model in models.items():
    model.fit(X_train_pca, y_train)

    y_scores = model.predict_proba(X_test_pca)[:, 1]
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_scores)
    auc_pr = auc(recall_curve, precision_curve)

    plt.plot(recall_curve, precision_curve, label=f'{model_name} (AUC-PR = {auc_pr:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.show()

individual_predictions = []

for model_name, model in models.items():
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    individual_predictions.append(y_pred)

individual_predictions = np.array(individual_predictions)

ensemble_predictions = np.mean(individual_predictions, axis=0) > 0.5  

ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
print(f"Ensemble Model (Majority Voting) Accuracy: {ensemble_accuracy:.2f}")


# In[ ]:


models = {
   'Logistic Regression': LogisticRegression(C= 100, penalty= 'l1', solver= 'liblinear'),
    'Random Forest': RandomForestClassifier(max_depth= None, min_samples_split= 5, n_estimators= 100),
    'Decision Tree': DecisionTreeClassifier(max_depth= 3, min_samples_leaf= 2, min_samples_split= 2),
    'KNN': KNeighborsClassifier(n_neighbors= 7, p= 2, weights ='uniform'),
}

plt.figure(figsize=(20, 5))

def logloss_train(train_pca,y):
    y_train_prob = model.predict_proba(train_pca)[:, 1]
    return log_loss(y, y_train_prob)

def logloss_test(test_pca,y):
    y_test_prob = model.predict_proba(test_pca)[:, 1]
    return log_loss(y, y_test_prob)

for model_name, model in models.items():
    model.fit(X_train_pca, y_train)

    log_loss_train = logloss_train(X_train_pca,y_train)

    log_loss_test = logloss_test(X_test_pca, y_test)
    print(f"Model: {model_name}")
    print(f"Log Loss (Training): {log_loss_train:.4f}")
    print(f"Log Loss (Testing): {log_loss_test:.4f}")

    plt.bar(model_name + ' (Training)', log_loss_train, color='blue')
    plt.bar(model_name + ' (Testing)', log_loss_test, color='orange')

plt.ylabel('Log Loss')
plt.title('Log Loss for Classification Models')
plt.show()


# In[ ]:




