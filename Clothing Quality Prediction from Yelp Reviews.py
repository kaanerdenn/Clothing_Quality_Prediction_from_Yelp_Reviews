import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

cloth = pd.read_json("cloth_yelp.json", lines=True)

cloth['quality'].value_counts()

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Describe #####################")
    print(dataframe.describe())
    print("##################### Info #####################")
    print(dataframe.info())
check_df(cloth, head=2)


# Removing columns having too much null values first.
cloth = cloth.drop(['item_id', 'waist'], axis = 1)
cloth.head(2)

# Removing columns which wont provide much information.
cloth = cloth.drop(['user_name', 'user_id', 'review_summary',   'review_text'], axis = 1)
cloth.head(2)

new_cloth = cloth.dropna(how = 'any')
new_cloth.shape

# Finding categorical data:
new_cloth['cup size'].value_counts()
new_cloth['category'].value_counts()
new_cloth['length'].value_counts()
new_cloth['fit'].value_counts()

# Converting the categorical data:

from sklearn.preprocessing import LabelEncoder
def label_encoded(feat):
    le = LabelEncoder()
    le.fit(feat)
    print(feat.name,le.classes_)
    return le.transform(feat)


name_list = ['cup size', 'category', 'length', 'fit', 'shoe width']
for name in name_list:
  new_cloth[name] = label_encoded(new_cloth[name])
new_cloth.head(3)

clothes_copy = new_cloth.copy()


def ref1(x):
  vari = np.nan
  try:
    vari = float(x)
  except:
    vari = np.nan
  return vari

clothes_copy['bust'] = clothes_copy['bust'].map(ref1)

clothes_copy['bust'].isnull().sum()

clothes_copy = clothes_copy.dropna(how = 'any')

import re
lis = clothes_copy['height'][:5].to_list()
print(lis)
k = []
p = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'
if re.search(p, lis[0]) is not None:
    for catch in re.finditer(p, lis[0]):
        k.append(int(catch[0]))

k

# Trying to extract data from height column.
height_list = clothes_copy['height'].to_list()
updated = []

def extractSize(x):
  numbers = []
  p = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'
  if re.search(p, x) is not None:
    for catch in re.finditer(p, x):
        numbers.append(int(catch[0]))
  # print(len(numbers))
  if len(numbers) == 2:
    fina = numbers[1] + numbers[0]*12
  elif len(numbers) == 1:
    fina = numbers[0] * 12
  else:
    fina = np.nan
  return fina

for height in height_list:
  updated.append(extractSize(height))


clothes_copy.head()

clothes_copy['height_inches'].isnull().sum()

sns.set_theme(style="whitegrid")
sns.boxplot(clothes_copy['size'])

sns.set_theme(style="whitegrid")
sns.boxplot(clothes_copy['height_inches'])


# clothes_copy = clothes_copy[clothes_copy['height_inches'] > 57]
# clothes_copy = clothes_copy[clothes_copy['height_inches'] < 74]
# clothes_copy = clothes_copy[clothes_copy['hips'] < 53]
# clothes_copy = clothes_copy[clothes_copy['bra size'] < 40]
# clothes_copy = clothes_copy[clothes_copy['bra size'] > 30]
# clothes_copy = clothes_copy[clothes_copy['bust'] > 24]
# clothes_copy = clothes_copy[clothes_copy['bust'] < 48]

clothes_copy.shape

plt.figure(figsize=(10, 10))
sns.heatmap(clothes_copy.corr(),annot=True,cmap='viridis',linewidths=.5)


y = clothes_copy['quality']
X = clothes_copy.drop(['quality', 'height', 'hips', 'bust'], axis = 1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state = 42)

random_model = RandomForestClassifier(n_estimators=250, n_jobs = -1)

#Fit
random_model.fit(Xtrain, ytrain)

#Checking the accuracy
random_model_accuracy = round(random_model.score(Xtrain, ytrain)*100,2)
print(round(random_model_accuracy, 2), '%')


random_model_accuracy1 = round(random_model.score(Xtest, ytest)*100,2)
print(round(random_model_accuracy1, 2), '%')


