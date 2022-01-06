import csv
import numpy as np
import sys


class nn():

    def __init__(self, train, validation, train_out, validation_out, metrics_out, num_epochs, hidden, init, learn_rate):
        
        self.train = train
        self.validation = validation

        self.train_out = train_out
        self.validation_out = validation_out
        self.metrics_out = metrics_out
        self.epochs = int(num_epochs)

        self.hidden = int(hidden)
        self.init = int(init)
        self.learn_rate = float(learn_rate)
        self.classes = 10
        self.size_feature = None
        self.size = None

        self.alpha = None
        self.beta = None

        self.input = None
        self.output = None


    def dataInputs(self, file_in):
        file = open(file_in)
        rows = file.readlines()
        file.close()
        lables = []
        features = []
        

        for row in rows: 
    
            elements = row.split(",")
            elements[-1] = elements[-1].rstrip()
            lable_vec = np.zeros(10)
            lable_vec[int(elements[0])] = 1
            
            lables.append(lable_vec)
            elements_b = elements[1:]
            elements_b = [1]+ elements[1:]
            features.append(elements_b)
            
            
        features = np.array(features)
        features = features.astype('int')
        self.size_feature = len(elements)-1

        return lables, features

    def init_weights(self, hidden, size_feature, classes, init):
        
        if init == 1:
            self.alpha = np.random.uniform(-0.1, 0.1, (hidden, size_feature+1))
            self.alpha[:, 0] = 0
            self.beta = np.random.uniform(-0.1, 0.1, (classes, hidden+1))
            self.beta[:, 0] = 0
        else:
            self.alpha = np.zeros((hidden, size_feature+1))
            self.beta = np.zeros((classes, hidden+1))

        return self.alpha, self.beta


    def forward(self, x, i):
        self.input = x[i]
        a = np.dot(self.alpha, x[i])
        z = 1/(1+np.exp(-a))
        z = np.append(1, z)
        b = np.dot(self.beta, z)
        self.output = b
        return self.output

    def loss(self, x, y, i):
        loss = 0
        
        exp_prob = np.exp(self.forward(x, i))/sum(np.exp(self.forward(x, i)))
        loss += np.dot(y[i], np.log(exp_prob))
        SGD_loss = -loss
        return SGD_loss
             

    def ybackward(self, x, y, i):
        loss = self.loss(x, y, i)
        exp_prob = np.exp(self.forward(x,i))/sum(np.exp(self.forward(x,i)))

        derivative_y = -y[i]/exp_prob

        return derivative_y

    def bbackward(self, x, y, i):
        exp_prob = np.exp(self.forward(x, i))/sum(np.exp(self.forward(x, i)))

        derivative_b = -y[i]+exp_prob

        return derivative_b

    def linear_backward(self, inputs, weights, gradient_comb):
        inputs = inputs.reshape((1, -1))
        gradient_comb = gradient_comb.reshape((-1, 1))
       
        gradient_weights = np.dot(gradient_comb, inputs)
  
        tran_weights = np.transpose(weights[:, 1:])
        gradient_inputs = np.dot(tran_weights, gradient_comb)
        return gradient_weights, gradient_inputs

    def sigmoid_backward(self, z, grad_z):
        z = z[1:]
        grad_z = grad_z.ravel()
        grad_a = np.multiply(np.multiply(grad_z, z), 1 - z)
        grad_a = grad_a.reshape(-1, 1)
        return grad_a

    def training(self, train, validation, train_out, validation_out, metrics_out, num_epochs, hidden, init, learn_rate):
        train_y, train_x = self.dataInputs(train)
        valid_y, valid_x = self.dataInputs(validation)
        self.alpha, self.beta = self.init_weights(self.hidden, self.size_feature, self.classes, self.init)
        metrics_out_file = open(metrics_out,"w")
        train_out_file = open(train_out,'w')
        test_out_file = open(validation_out,"w")
	
        for epoch in range(self.epochs):
            for i in range(len(train_x)):
                
                exp_prob = np.exp(self.forward(train_x,i))/sum(np.exp(self.forward(train_x,i)))
                a = np.dot(self.alpha, train_x[i])
                z = 1/(1+np.exp(-a))
                z = np.append(1, z)
                grad_b = self.bbackward(train_x, train_y, i)
                grad_beta, grad_z = self.linear_backward(z, self.beta, grad_b)
                grad_a = self.sigmoid_backward(z, grad_z)
                grad_alpha, grad_x = self.linear_backward(train_x[i], self.alpha, grad_a)
                
                self.alpha -= self.learn_rate * grad_alpha
                self.beta -= self.learn_rate * grad_beta

            train_loss = 0

            for i in range(len(train_x)):
                
                train_loss += self.loss(train_x, train_y, i)

            train_loss /= len(train_x)
            metrics_out_file.write("epoch={} crossentropy(train): {}\n".format(epoch + 1, train_loss))
            
            test_loss = 0
            for i in range(len(valid_x)):
                test_loss += self.loss(valid_x, valid_y, i)
                
            test_loss /= len(valid_x)
            metrics_out_file.write("epoch={} crossentropy(validation): {}\n".format(epoch + 1, test_loss))


        print(self.alpha)
        print(self.beta)

        error = 0
        for i in range(len(train_x)):
            pred = np.argmax(np.exp(self.forward(train_x, i))/sum(np.exp(self.forward(train_x, i))))
            
            train_out_file.write("{}\n".format(pred))
            error += (np.argmax(train_y[i]) != pred)
        metrics_out_file.write("error(train): {}\n".format(error / len(train_x)))

        error = 0
        for i in range(len(valid_x)):
            pred = np.argmax(np.exp(self.forward(valid_x, i))/sum(np.exp(self.forward(valid_x, i))))
            test_out_file.write("{}\n".format(pred))
            error += (np.argmax(valid_y[i]) != pred)
        metrics_out_file.write("error(validation): {}\n".format(error / len(valid_x)))

    def run(self):
        self.training(self.train, self.validation, self.train_out, self.validation_out, self.metrics_out, self.epochs, self.hidden, self.init, self.learn_rate)
                
        
def main():
    train = sys.argv[1]
    validation = sys.argv[2]
    train_out = sys.argv[3]
    validation_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epochs = sys.argv[6]
    hidden = sys.argv[7]
    init = sys.argv[8]
    learn_rate = sys.argv[9]

    model = nn(train, validation, train_out, validation_out, metrics_out, num_epochs, hidden, init, learn_rate)
    model.run()


if __name__ == '__main__':
    main()



    
            
                              
                        



        
