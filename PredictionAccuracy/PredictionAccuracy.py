# import dataset
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
Y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .5) 

from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(X_train, Y_train)

from sklearn.neighbors import KNeighborsClassifier
my_classifier2 = KNeighborsClassifier()
my_classifier2.fit(X_train, Y_train)

predictions = my_classifier.predict(X_test)
print predictions

predictions2 = my_classifier2.predict(X_test)
print predictions2

from sklearn.metrics import accuracy_score
print "Accuracy by DecisionTree is %f" % accuracy_score(Y_test, predictions)
print "Accuracy by KNeighbors is %f" % accuracy_score(Y_test, predictions2)
