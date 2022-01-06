# Author: Qian ZHANG
# Assignment 2 - Inspecting the Data

import sys
import numpy as np
import csv
from collections import Counter
import math

class Node():
    def __init__(self, train_in):
        self.train_data = np.loadtxt(train_in, dtype=str, delimiter = '\t')
        self.value = [] # store the values for Y
        self.entropy = 0 
        self.train_error = 0
        self.pred = None

    def train(self):
        
        for i in range(len(self.train_data)):
            value = self.train_data[i][-1]
            self.value.append(value)
            self.pred=self.majority(self.value)
            
    

    def compute_majority(self, data):
        distribution = Counter(data)
        for most, num_times in distribution.most_common(1):
            p = num_times/(len(data)-1)
            self.entropy = -p*math.log(p, 2)-(1-p)*math.log(1-p, 2)
            return self.entropy

    def majority(self, data): 
        common = Counter(data)
        for most, num_times in common.most_common(1):
            return most

    def ErrorRate(self, data):
        distribution = Counter(data)
        for most, num_times in distribution.most_common(1):
            error = float((len(data)-num_times-1))/(len(data)-1)
            return error
    
    def outputErrors(self, filename):
        file = open(filename, 'w')
        file.write("entropy: %f\n" % self.entropy)
        file.write("error: %f\n" % self.train_error)
        file.close()
        
def main():
    train_data = sys.argv[1]
    metrics_output = sys.argv[2]
    
    Stump = Node(train_data)
    Stump.train()

    Stump.entropy = Stump.compute_majority(Stump.value)
    Stump.train_error = Stump.ErrorRate(Stump.value)
    Stump.outputErrors("%s" % metrics_output)


if __name__ == '__main__':
    main()



        
