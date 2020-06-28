## Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sweetviz
from category_encoders.one_hot import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


## Reading the File and Checking for null values
df = pd.read_csv('student-mat.csv',delimiter= ';')
# print(df.dtypes)
# print(df.isnull().values.any())


## Analysing the Data
# my_report = sweetviz.analyze([df,'Train'], target_feat= 'G3')
# my_report.show_html()


## Scaling and Encoding the data
for colum in df.columns:
    if df[colum].dtype == object:
        df[colum] = OneHotEncoder().fit_transform(df[colum])

df = MinMaxScaler().fit_transform(df)
df = pd.DataFrame(df,columns=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2','G3'])

# print(df.head())
# print(df.dtypes)


## Finding the Correlations between Features
# sns.heatmap(df.corr(), fmt = '.1f',annot = True)
# plt.show()

correlations = df.corr()['G3'].drop('G3')
#print(correlations)
# print(correlations.quantile(.25))
# print(correlations.quantile(.75))


## Choosing the best threshold for improving the model
def get_features(correlation_threshold):
    abs_corrs = correlations.abs()
    high_correlations = abs_corrs[abs_corrs > correlation_threshold].index.values.tolist()
    return high_correlations

# thresh = []
# scores = []
# for i in np.arange(start = 0.04,stop = 0.11,step = 0.01):
#     features = get_features(i)
#     thresh.append((i))
#     X = df[features]
#     Y = df.G3
#
#     x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=4)
#     classifier = LinearRegression()
#     classifier.fit(x_train, y_train)
#     score = classifier.score(x_test, y_test)
#     scores.append(score)
#
# plt.plot(thresh,scores)
# plt.xlabel('thrshold_values')
# plt.ylabel('scores')
# plt.show()


## Final Threshold with greatest Score
features = get_features(0.06)
# print(len(features))
X = df[features]
Y = df.G3


## Spliting, Fiting, getting the score of Model
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=4)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
print("R2 Score of the regression model is :-",regressor.score(x_test, y_test))