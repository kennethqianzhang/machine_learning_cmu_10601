### Assignment 4
### Qian ZHANG

import sys
import numpy as np
import csv

class featureEngineering():
    def __init__(self, train, validation, test, dictionary, train_out, validation_out, test_out, flag):
        self.train = train
        self.validation  = validation
        self.test = test
        self.train_out = train_out
        self.validation_out = validation_out
        self.test_out = test_out

        self.dictionary = dictionary
        self.flag = int(flag)
        self.threshold = 4

    def featureEngi(self, file_in, file_out):
        with open(file_in) as dat:
        
            dic = dict()
            rd = csv.reader(dat, delimiter = "\t")
        
            dct = open("dict.txt")
            refs = dct.readlines()
            dct.close()
            for ref in refs:
                key, val = ref.split()
                dic[key] = val
            
            ys = []
            features = []
            
            for row in rd:
                ys.append(row[0])
                line = row[1]
                words = line.split()
                feature = dict()
                for word in words: 
                    if word in dic.keys():
                        if dic[word] not in feature:
                              feature[dic[word]] = 1
                        elif dic[word] in feature:
                            feature[dic[word]] += 1
                features.append(feature)
                
            self.filesOutput(ys, features, file_out)

    def filesOutput(self, ys, features, file_out):
        file = open(file_out, 'w')
        for i in range(len(ys)):
            output = ""
            output += "%s" % (ys[i])
            current_ref = features[i]
            for word in current_ref.keys():
                if self.flag == 1:
                    output += "\t%s:1" % (word)
                elif self.flag == 2:
                    if current_ref[word] < self.threshold:
                        output += "\t%s:1" % (word)
            output += "\n"
            file.write(output)
        file.close()


    def allDatasets(self):
        self.featureEngi(self.train, self.train_out)
        self.featureEngi(self.validation, self.validation_out)
        self.featureEngi(self.test, self.test_out)

def main():
    train = sys.argv[1]
    validation = sys.argv[2]
    test = sys.argv[3]
    dictionary = sys.argv[4]
    train_out = sys.argv[5]
    validation_out = sys.argv[6]
    test_out = sys.argv[7]
    flag = sys.argv[8]

    model = featureEngineering(train, validation, test, dictionary, train_out, validation_out, test_out, flag)
    model.allDatasets()

if __name__ == '__main__':
    main()
    
