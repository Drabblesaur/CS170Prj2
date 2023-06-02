import numpy as np

class NNClassifier:
    def __init__(self,data,K) -> None:
        self.data = data
        #get the first column and set to y
        self.y = data[:,0]
        #get the rest of the columns and set to X
        self.X = data[:,1:]
        #n and d are the number of rows and columns in X
        self.n = self.X.shape[0]
        self.d = self.X.shape[1]
        #K is the number of nearest neighbors to use
        self.K = K
    
    # Predict the class of a single point x
    def predict(self,x):
        #calculate the euclidian distance between x and all the points in X
        dist = np.linalg.norm(self.X-x,axis=1)
        #sort the distance where y is the same from closest to furthest
        sorted_index = np.argsort(dist)
        #print(sorted_index)
        #print(self.y[sorted_index])
    
        #get the first K elements and their y values
        K_nearest = self.y[sorted_index[:self.K]]
        #given the K nearest neighbors, predict the class of x
        classes,counts = np.unique(K_nearest,return_counts=True)
        print(classes)
        print(counts)
        #return the class with the most counts
        return classes[np.argmax(counts)]
    
    # Predict the class of all points in X
    def predict_all(self,X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_pred[i] = self.predict(X[i,:])
        return y_pred

    

        



class Validator:
    def __init__(self,data,Classifier,feature_set) -> None:
        self.data = data
        self.Classifier = Classifier
        self.feature_set = feature_set
        pass

    def leave_one_out(self):
        n = self.data.shape[0]
        y_pred = np.zeros(n)
        for i in range(n):
            train_data = np.delete(self.data,i,axis=0)
            test_data = self.data[i,:]
            Classifier = self.Classifier(train_data,self.feature_set)
            y_pred[i] = Classifier.predict(test_data[:-1])
        return y_pred
    

def main():
    filename = "small-test-dataset.txt"
    data = np.loadtxt(filename, usecols=range(11))
    test_point = data[0]
    #remove the first row
    data = np.delete(data,0,axis=0)
    #print(data.shape)
    classifier = NNClassifier(data,6)
    y_pred = classifier.predict(test_point[:-1])
    print("PREDICTED CLASS:",y_pred)


main()