from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from pandas import Series
import pandas as pd

data = []
data = pd.read_csv('~/russiadata.newversion.csv', sep = ',', low_memory= False)
data.head(1).copy()
interests_col = [col for col in data.columns if 'Interests' in col]
x = data.loc[:, interests_col]
x = x.stack()
z = pd.DataFrame(Series(pd.Categorical(x[x!=0].index.get_level_values(1))))
z = pd.DataFrame(z)
a = []
for i in range(len(z)):
    a.append(z.iloc[i,0].replace('Interests:.', ''))

from keras.preprocessing.text import hashing_trick
#function to strip all @,# and entities (&,^,<,>, etc) but keeps the "'" (for They're, It's etc)
def strip_entities(text):
    import string
    # Put entites you dont want removed in entity_prefixes
    entity_prefixes = []
    for separator in string.punctuation:
        if separator not in entity_prefixes:
            text = text.replace(separator,'')
    words = []
    for word in text.split():
        word = word.strip()
        #Now check if word is still a word?
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

import re
for i in range(len(a)):
    a[i] = strip_entities(a[i])
    a[i] = re.sub(' ','', a[i])

b = []

for i in range(len(z)):
    b.append(hashing_trick(a[i], round(len(set(a))*1.3), hash_function='md5'))

b = pd.DataFrame(b)
len(set(z[0]))
len(set(a))
len(set(b[0]))

import numpy as np
y = data.iloc[:, 6]
y = y[~np.isnan(y)]
X = data.iloc[:, 10:23]
X = X[~np.isnan(X)]
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import scipy

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.20)


estimator = RandomForestRegressor()

selector = RFE(estimator, n_features_to_select = 5 )
selector = selector.fit(X, y)
selector.ranking_
selector.support_
selector.estimator_
model = estimator.fit(x_train.iloc[:,[1,3,5,8,10]],y_train)
scores = cross_val_score(estimator, x_train.iloc[:,[1,3,5,8,10]],y_train, cv = 5)
scores.mean()
model.score(x_test.iloc[:,[1,3,5,8,10]],y_test)

test = pd.DataFrame( data = {'PredictedValues' : model.predict(x_test.iloc[:,[5,10]]), 'ActualValues' : y_test} )

from sklearn.metrics import r2_score
r2_score(test.loc[:,'ActualValues'], test.loc[:,'PredictedValues'])

x_test.iloc[:,[0,1]]

y_test

