### 10-601 Homework 1
### Qian ZHANG

import sys
import numpy as np
from collections import Counter


class Node():
    def __init__(self, train_data, test_data, split_ind):
        # load train_data and test_data 
        self.train_data = np.loadtxt(train_data, dtype=str, delimiter='\t') 
        self.test_data = np.loadtxt(test_data, dtype=str, delimiter='\t')
        self.split_ind = int(split_ind)
        # Create empty values and results
        self.values = [] 
        self.results = []
        # list of results from one of the split sides
        self.category1 = [] 
        self.category2 = []
        
        self.category1_result = None
        self.category2_result = None

        # prediction values for train and test
        self.train_predict = [] 
        self.test_predict = []

        # error rate calculation
        self.train_error = 0 
        self.test_error = 0



    # Seperate self.values[0] to either self.category1_result or self.category2_result
    def train(self): 

        
        # split data into category 1 and category 2
        
        for i in range(len(self.train_data)): 
            

            # Taking ith row and the threshold number self.split_ind
            # result means the actual 
            
            val = self.train_data[i][self.split_ind]
            result = self.train_data[i][-1]

            if i == 0:  continue
            if i == 1: 
                self.values.append(val)
                self.category1.append(result)
                self.results.append(result)

            else:

                if val not in self.values:
                    self.values.append(val)
                    self.category2.append(result)
                    
                    
                    if result not in self.results:
                        self.results.append(result)

                else:
                    if self.values.index(val) ==0:
                        self.category1.append(result)
                        if result not in self.results:
                            self.results.append(result)
                    else:
                        self.category2.append(result)
                        if result not in self.results:
                            self.results.append(result)
        
            
            
        # if the majority (more than half) is category 1, then belongs to 1
        
        self.category1_result = self.majority(self.category1)
        if self.category1_result == self.results[0]: 
            self.category2_result = self.results[1]
        else:
            self.category2_result = self.results[0]


    def test(self, data, filename, predictions):
        file = open(filename, 'w')
        for i in range(len(data)):
            if i == 0: continue
            val = data[i][self.split_ind]
            if val == self.values[0]:
                result = self.category1_result
            else:
                result = self.category2_result
            predictions.append(result)
            file.write("%s\n" %result)
        file.close()

    def ErrorRate(self, data, predictions):
        error_count = 0
        for i in range(len(predictions)):
            if predictions[i] != data[i+1][-1]:
                error_count += 1
        error = error_count / (len(data)-1)
        return error
    
    def outputErrors(self, filename):
        file = open(filename, 'w')
        file.write("error(train): %f\n" % self.train_error)
        file.write("error(test): %f\n" % self.test_error)
        file.close()

 # returns the majority vote given a dataset

    def majority(self, data): 
        common = Counter(data)
        for most, num_times in common.most_common(1):
            return most

def main():
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    split_ind = sys.argv[3]
    train_pred = sys.argv[4]
    test_pred = sys.argv[5]
    metrics_output = sys.argv[6]

    """
    ### General Form Requirement
    
    train_file = '?_'+'train.tsv'
    test_file = '?_'+'test.tsv'
    split_ind = x
    train_pred = '?_' + str(split_ind) + '_train.labels'
    test_pred = '?_' + str(split_ind) + '_test.labels'
    metrics_output = '?_' + str(split_ind) + '_metrics.txt'

    """

    Stump = Node(train_data, test_data, split_ind)
    Stump.train()
    Stump.test(Stump.train_data, "%s" % train_pred, Stump.train_predict)
    Stump.test(Stump.test_data, "%s" % test_pred, Stump.test_predict)

    Stump.train_error = Stump.ErrorRate(Stump.train_data, Stump.train_predict)
    Stump.test_error = Stump.ErrorRate(Stump.test_data, Stump.test_predict)
    Stump.outputErrors("%s" % metrics_output)


if __name__ == '__main__':
    main()
