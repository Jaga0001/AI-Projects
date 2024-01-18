# Loading datasets

from pandas import read_csv

fileName = 'Energy Meter.csv'
names = ['Voltage', 'Current', 'Power', 'Class']
dataset = read_csv(fileName, names=names)

# Summarizing the Dataset
print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())
print(dataset.groupby("Class").size())

# Data Visualization
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

dataset.plot(kind='bar', subplots=True, layout=(2, 2), sharex=False, sharey=False)
pyplot.title('Bar Chart')
pyplot.show()

dataset.hist()
pyplot.title('Histogram')
pyplot.show()

scatter_matrix(dataset)
pyplot.title('Scatter Matrix')
pyplot.show()

# Machine Learning Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

# Splitting the dataset
array = dataset.values
X = array[:, 0:3]
Y = array[:, 3]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1, shuffle=True)

# Spot-Checking Algorithms
models = [('LR', LogisticRegression(solver='liblinear', multi_class='ovr')), ('LDA', LinearDiscriminantAnalysis()),
          ('KNN', KNeighborsClassifier()), ('CART', DecisionTreeClassifier()), ('NB', GaussianNB()),
          ('SVM', SVC(gamma='auto'))]

results = []
names = []
res = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    res.append(cv_results.mean())
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

pyplot.ylim(0.990, 0.999)
pyplot.bar(names, res, color='blue')
pyplot.title('Algorithm Comparison')
pyplot.show()
