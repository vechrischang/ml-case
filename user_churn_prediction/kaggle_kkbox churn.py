
# coding: utf-8

# # Goal
# ***In this task, you will be predicting whether a user will churn after their subscription expires. Specifically, we want to see if a user make a new service subscription transaction within 30 days after their current membership expiration date.***

# In[2]:


from __future__ import division
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN

    # trsc data
df_transaction = pd.read_csv('transactions_v2.csv', error_bad_lines=False)
df_transaction.head()


# In[3]:


# churn_data
df_submission = pd.read_csv('sample_submission_v2.csv')
df_train = pd.read_csv('train_v2.csv')


# In[4]:


# combin sub x train
df_is_chrun = [df_submission,df_train]
df_is_chrun = pd.concat(df_is_chrun)
df_is_chrun['is_churn'].sum()


# In[ ]:


# log data
df_log = pd.read_csv('user_logs_v2.csv')
df_log.head()


# In[5]:


# user data
df_member = pd.read_csv('members_v3.csv')
df_member.head()


# In[6]:


# merge
df_merge = pd.merge(df_member, df_is_chrun, how= 'left', on = 'msno')
df_merge = pd.merge(df_merge, df_transaction, how= 'left', on = 'msno')
df_merge.head()


# In[7]:


# sql like distonct
df_merge = df_merge.drop_duplicates()
# remove nan
df_merge = df_merge.dropna()
df_merge.head()


# In[8]:


# transfer gender
df_merge['sex'] = np.where(df_merge['gender'] == 'male',1,0)
# change metric type
df_merge[['is_churn']] = df_merge[['is_churn']].astype(int)
print(df_merge.dtypes)


# In[9]:


# Isolate y
y = df_merge['is_churn']
# drop useless cols
drop_cols = ['city','bd','registered_via','msno', 'gender','registration_init_time',
         'transaction_date','membership_expire_date','is_churn','is_auto_renew','is_cancel','sex']
df_merge_drop = df_merge.drop(drop_cols,axis=1)
df_merge_drop = df_merge_drop.dropna() 
df_merge_drop.head()


# In[10]:


# reset rows order
df_merge_drop = df_merge_drop.reset_index()
#df_merge_drop = df_merge_drop.apply(lambda x: pd.to_numeric(x,errors='ignore'))
print(df_merge_drop.isnull().any())
#df_merge_drop = df_merge_drop.fillna(lambda x: x.median())
df_merge_drop = df_merge_drop.fillna(method='ffill')
print('---')
df_merge_drop.head()


# In[11]:


# Pull out features for future use
features = df_merge_drop.columns
print(features)


# In[12]:


# define X
X = df_merge_drop.as_matrix().astype(np.float32)

# Standardize features by removing the mean and scaling to unit variance    
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.preprocessing import Imputer

scaler = preprocessing.StandardScaler()
#scaler = StandardScaler()
X = scaler.fit_transform(X)

imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
X = imp.fit_transform(X)


print('Feature space holds %d observations and %d features' % X.shape)
print("Unique target labels:", np.unique(y))
print(X.dtype)


# In[260]:


def is_finite(df):
    print(df[:,0].shape)
    index = 0
    for i in df[:,0]:
        if not np.isfinite(i):
            print(index, i)
        index +=1
is_finite(X)
print('---')

def _assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)
print(_assert_all_finite(X))
print('---')
print(np.any(np.isnan(X)))
print('---')
print(np.all(np.isfinite(X)))


# In[ ]:


# clean the dataset of nan, Inf, and missing cells
def clean_dataset(df):
assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
df.dropna(inplace=True)
indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
return df[indices_to_keep].astype(np.float32)

clean_dataset(df_merge_drop)
print(df_merge_drop.dtypes)


# In[ ]:


from sklearn.cross_validation import KFold

def run_cross_validation(X,y,clf_class,**kwargs):
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_pred=y.copy()
    
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(x_test)
        return y_pred


# In[ ]:


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN

def accuracy(y_true,y_pred):
    # NumPy interprets True and False as 1. and 0.
    return np.mean(y_true == y_pred)

print("svm:")
print("%.3f" % accuracy(y, run_cross_validation(X,y,SVC)))
print("rf:")
print("%.3f" % accuracy(y, run_cross_validation(X,y,RF)))
print("knn:")
print("%.3f" % accuracy(y, run_cross_validation(X,y,KNN)))


# ### svm

# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


clf = SVC()
clf.fit(X_train, y_train)


# In[17]:


from sklearn import metrics
from sklearn.cross_validation import cross_val_score

y_pred = clf.predict(X_test)

accurate_score = metrics.accuracy_score(y_test, y_pred) #np.mean(y_test == y_pred)
#scores = cross_val_score(clf, y_pred, y_test, cv=10, scoring='accuracy')
print(accurate_score)
print('---')
#print(scores.mean())


# In[23]:


from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, 
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# In[ ]:


# AUC & ROC

