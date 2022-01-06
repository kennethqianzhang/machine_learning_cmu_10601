import numpy as np
import csv
import math
import sys

class GaussianNB():

    def __init__(self, train, test, train_out, test_out, metrics_out, voxels):

        self.train = train
        self.test = test

        self.train_out = train_out
        self.test_out = test_out
        self.metrics_out = metrics_out
        self.voxels = voxels



    def dataInputs(self, file_in):
        file = open(file_in)
        rows = file.readlines()

        file.close()
        building = []
        tool = []
        data = []
        data_label = []

        for row in rows:
            elements = row.split(",")
            if elements[-1] == "building\n":
                building.append(elements[:-1])

            elif elements[-1] == "tool\n":
                tool.append(elements[:-1])
        for row in rows[1:]:
            elements = [float(x) for x in row.split(",")[:-1]]
            data.append(elements)
            data_label.append(row.split(",")[-1].rstrip())

        p_label_tool = data_label.count("tool")/len(data_label)
        p_label_building = 1- p_label_tool

        return building, tool, data, data_label, p_label_tool, p_label_building


    def training(self):
        building, tool, data, data_label, p_label_tool, p_label_building = self.dataInputs(self.train)
        mu_building = []
        sigma_building = []
        mu_tool = []
        sigma_tool = []

        for col in range(len(building[0])):
            building_data = []
            sq_data = []
            for row in range(len(building)):
                building_data.append(float(building[row][col]))
            mu = sum(building_data)/len(building)
            mu_building.append(mu)

            for i in range(len(building_data)):
                sq_data.append(pow(building_data[i]-mu, 2))

            sigma = sum(sq_data)/len(building)
            sigma_building.append(sigma)

        for col in range(len(tool[0])):   
            tool_data = []
            sq_data = []
            for row in range(len(tool)):
                tool_data.append(float(tool[row][col]))
        
            mu = sum(tool_data)/len(tool)
    
            mu_tool.append(mu)
    
            for i in range(len(tool_data)):
                sq_data.append(pow(tool_data[i]-mu,2))
        
            sigma = sum(sq_data)/len(tool)
    
            sigma_tool.append(sigma)


        return mu_building, sigma_building, mu_tool, sigma_tool
    

    def predict(self, inputs):
        building, tool, data, data_label, p_label_tool, p_label_building = self.dataInputs(inputs)
        mu_building, sigma_building, mu_tool, sigma_tool = self.training()

        prediction = []
        error = 0

        diff = []
        for i in range(len(mu_building)):
            diff.append(abs(mu_building[i] - mu_tool[i]))
        top = np.argsort(diff)
        
        

        for row in range(len(data)):
            p_building_set = []
            p_tool_set = []

            for col in top[-int(self.voxels):]:
        
                p_building_set.append(math.log(math.exp(-pow((data[row][col]-mu_building[col]),2)/(2*sigma_building[col]))/math.sqrt(2*math.pi*sigma_building[col])))
                p_tool_set.append(math.log(math.exp(-pow((data[row][col]-mu_tool[col]),2)/(2*sigma_tool[col]))/math.sqrt(2*math.pi*sigma_tool[col])))
            p_building = math.log(p_label_building) + sum(p_building_set)
            p_tool = math.log(p_label_tool) + sum(p_tool_set)
     
            if p_building > p_tool:
                prediction.append("building")
            else:
                prediction.append("tool")

        for i in range(len(prediction)):
            if prediction[i] != data_label[i]:
                error += 1
        error_rate = error/len(prediction)

        return error_rate, prediction

    def output(self):
        train_error, train_pred = self.predict(self.train)
        test_error, test_pred = self.predict(self.test)
        train_out_file = open(self.train_out,'w')
        test_out_file = open(self.test_out,"w")
        file = open(self.metrics_out, 'w')
        file.write("error(train): %f\n" %(train_error))
        file.write("error(test): %f\n" %(test_error))
        for i in range(len(train_pred)):
            train_out_file.write("{}\n".format(train_pred[i]))
                                 
        for i in range(len(test_pred)):
            test_out_file.write("{}\n".format(test_pred[i]))
        train_out_file.close()
        test_out_file.close()
        file.close()
        
        

def main():
    train = sys.argv[1]
    test = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    voxels = sys.argv[6]

    model = GaussianNB(train, test, train_out, test_out, metrics_out, voxels)
    model.output()


if __name__ == '__main__':
    main()
        

        
                
        
