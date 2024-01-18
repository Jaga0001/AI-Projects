from sklearn.svm import SVC
from pandas import read_csv
from sklearn.model_selection import train_test_split
import pickle
fileName = 'Energy Meter.csv'
names = ['Voltage', 'Current', 'Power', 'Class']
dataset = read_csv(fileName, names=names)

array = dataset.values
X = array[:, 0:3]
Y = array[:, 3]
X_train, X_Val, Y_train, Y_Val = train_test_split(X, Y, test_size=0.20, random_state=1, shuffle=True)

model = SVC(gamma='auto')
model.fit(X_train, Y_train)

# Saving the model
file = "Model.pkl"
pickle.dump(model, open(file, 'wb'))
print("Model Saved Successfully")

# Loading the model
l_model = pickle.load(open(file, 'rb'))
print("Model Loaded Successfully")
results = l_model.score(X_Val, Y_Val)
print(results)

# Predicting the class
value = [[215.0313, 0.17388, 37.389642443999996]]
predictions = l_model.predict(value)
print(predictions[0])
