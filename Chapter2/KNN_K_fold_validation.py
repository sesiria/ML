
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# import iris data set
iris = datasets.load_iris()
X = iris.data
y = iris.target

# defin the k for KNN algorithm
parameters = {'n_neighbors' : [1, 3, 5, 7, 9, 11, 13, 15]}
knn = KNeighborsClassifier()  # ! do not specify any parameters for the constructor

# search the best value for K via GridSearchCV
clf = GridSearchCV(knn, parameters, cv = 5)
clf.fit(X, y)

# print the best score of K
print ("best score is: %.2f"%clf.best_score_, " best param: ", clf.best_params_)
