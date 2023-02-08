import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report



cols = ['length','width','size','conc','concl','asym','m3long','m3trans','alpha','dist','class']
df = pd.read_csv("./dataSets/magic04.data", names=cols)
df['class'] = (df['class'] == 'g').astype(int)
df.head()

#visualaizing our data 
# for label in cols[:-1]:
#     plt.hist(df[df['class']==1][label], color='blue', label='gamma', alpha=0.7, density=True)
#     plt.hist(df[df['class']==0][label], color='red', label='hadron', alpha=0.7, density=True)
#     plt.title(label)
#     plt.ylabel("Probability")
#     plt.xlabel(label)
#     plt.show()




#creating our datasets ( train, valid and testing )
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)),int(0.8*len(df))])



#it is better to normalize all of the data values
def scale_dataset(dataframe, oversample=False):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    #will take the x and fit the standard scaler to x and for transform all values of x
    x = scaler.fit_transform(x)

    #now, the number of rows having g = 1 is almost twice the number of rows having g = 0. So, we need to add more rows where g = 0
    #what it is doing is that it is taking the smaller class and make it bigger so it match the number of the bigger class.
    if(oversample):
        ros = RandomOverSampler()
        x,y = ros.fit_resample(x,y)


    #will make all data huge 2d array
    #takes two arrays and puts them side by side
    #x is already 2d array mean while y is array of values, so we need to reshapre it into 2D array
    data = np.hstack((x, np.reshape(y, (len(y), 1))))

    return data, x, y


#len before scaling and fitting
# print("Lengths before scaling and fitting")
# print(len(train[train['class']==1]))
# print(len(train[train['class']==0]))


train, train_x, train_y = scale_dataset(train, True)
valid, valid_x, valid_y = scale_dataset(valid, False)
test, test_x, test_y = scale_dataset(test, False)


# print("Lengths after scaling and fitting")
# print(sum(train_y == 1))
# print(sum(train_y == 0))



# implementing kNN
def kNN():
    knn_model = KNeighborsClassifier(n_neighbors=1)
    knn_model.fit(train_x, train_y)


    y_predictions = knn_model.predict(test_x)
    print("kNN Results")
    print(classification_report(test_y, y_predictions))


# Naive bayes
def nb():
    nb_model = GaussianNB()
    nb_model.fit(train_x, train_y)
    y_predictions = nb_model.predict(test_x)
    print("NB Results")
    print(classification_report(test_y, y_predictions))

# Log Regression
def log():
    log_model = LogisticRegression()
    log_model.fit(train_x, train_y)
    y_predictions = log_model.predict(test_x)
    print("Log Results")
    print(classification_report(test_y, y_predictions))



kNN()
nb()
log()