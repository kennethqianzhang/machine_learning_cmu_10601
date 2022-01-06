import sys
import numpy as np
import csv

class LR():
    def __init__(self, train, validation, test, dictionary, train_out, test_out, metrics_out, num_epoch):
        self.train = train
        self.validation = validation
        self.test = test
        self.dictionary = dictionary

        self.train_out = train_out
        self.test_out = test_out
        self.metrics_out = metrics_out
        self.epochs = int(num_epoch)

        self.learn_rate = 0.1

        self.train_ys, self.train_features = self.dataInputs(self.train)
        self.validation_ys, self.validation_features = self.dataInputs(self.validation)
        self.test_ys, self.test_features = self.dataInputs(self.test)
        self.weights = self.training(self.train_features, self.train_ys, self.epochs)


    def dataInputs(self, file_in):
        file = open(file_in)
        rows = file.readlines()
        file.close()
        ys = []
        features = []

        for row in rows:
            feature = []
            words = row.split("\t")
            ys.append(int(words[0]))
            x_list = words[1:]
            for x in x_list:
        
                value_x = x.split(":")[0]
                feature.append(int(value_x))
            features.append(feature)
            
        return ys, features


    def training(self, features, ys, epochs):
        weights = np.zeros(39177)
        for epoch in range(epochs):
            for i in range(len(ys)):
                xs = dict()
                for word in features[i]:
                    xs[word] = 1
                xs[39176] = 1

                objec = 0
                for word in xs.keys():
                    objec += xs[word]*weights[word]

                exp_objec = np.exp(objec)
                theta_j = dict()

                for j in xs.keys():
                    marginal = self.learn_rate * xs[j] * (ys[i] - (exp_objec/(1+exp_objec)))/len(ys)
                    theta_j[j] = marginal

                for word in xs.keys():
                    weights[word] += theta_j[word]

        return weights

    def predict(self,ys, features, weights, file_out):
        total = 0
        count = 0
        file = open(file_out, 'w')
        for i in range(len(ys)):
            xs = dict()
            for word in features[i]:
                xs[word] = 1
            xs[39176] = 1

            objec = 0
            for word in xs.keys():
                objec += xs[word]*weights[word]

            result = 1 / (1 + np.exp(-objec))

            if result >= 0.5:
                prediction = 1
            else:
                prediction = 0

            total += 1

            if prediction == ys[i]:
                count += 1

            file.write("%d\n"%(prediction))
        file.close
        return 1 - count/total

    def output(self):
        train_error = self.predict(self.train_ys, self.train_features, self.weights, self.train_out)
        test_error = self.predict(self.test_ys, self.test_features, self.weights, self.test_out)
        file = open(self.metrics_out, 'w')
        file.write("error(train): %f\n" %(train_error))
        file.write("error(test): %f\n" %(test_error))
        file.close()
            
    
def main():
    train = sys.argv[1]
    validation = sys.argv[2]
    test = sys.argv[3]
    dictionary = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    num_epoch = sys.argv[8]

    lr = LR(train, validation, test, dictionary, train_out, test_out, metrics_out, num_epoch)
    lr.output()


if __name__ == '__main__':
    main()
                    
                
                    

    

    

    
            
        
