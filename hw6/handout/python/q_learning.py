import sys
from environment import MountainCar
import numpy as np

class Qmodel:
    def __init__(self, mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, learning_rate):
        self.mode = mode

        self.weight_out = weight_out
        self.returns_out = returns_out
        self.episodes = episodes
        self.max_iter = max_iterations
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.car = MountainCar(self.mode)
        self.num_actions = 3
        self.num_states = self.car.state_space

        self.weights = np.zeros((self.num_states, self.num_actions))
        self.bias = 0

        self.done = False


    def learning(self):
        values = []
        weights = np.zeros((self.num_states, self.num_actions))
        bias = 0
        
        for i in range(self.episodes):
            
            state = self.car.reset()
            sum_reward = 0
            num_iter = 0
            if num_iter < self.max_iterations:
                
                Qvalue = np.array([state.dot(weights[action]) + bias for action in [0, 1, 2]])
                rand_value = random.uniform(0,1)

                if  rand_value <= 1-self.epsilon:
                    optimal_action =  np.argmax(Qvalue)
                else:
                    optimal_action = np.random.choice([0, 1, 2])

                new_state, reward, self.done = self.car.step(optimal_action)

                new_Q = np.array([new_state.dot(weights[action]) + bias for action in [0, 1, 2]])

                sum_reward += reward

                new_weight = np.zeros((self.num_states, self.num_actions))
                for j in state:
                    new_weight[int(j)][optimal_action] = state[j]

                discount = reward + self.gamma * np.max(new_Q)

                weights -= self.learning_rate *(Qvalue[action] - discount) * new_weight
                bias -= self.learning_rate * (Qvalue[action] -discount)

                state = new_state

                num_iter +=1

            values.append(sum_reward)

        self.weights = weights
        self.bias = bias

        print(self.bias)
        print(self.weights)
        print(values)

        return values

    def output(self):
        rewards = self.learning()
        returns_out = open(self.returns_out, 'w')

        for i in range(len(rewards)):
            returns_out.write("%f\n" %rewards[i])
        returns_out.close()

        weights_out = open(self.weight_out, 'w')
        weights_out.write("%f\n" %self.bias)
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                weights_out.write("%f\n" %self.weights[i][j])
        weights_out.close()

        


def main(args):
    mode = args[1]
    weight_out = args[2]
    returns_out = args[3]
    episodes = args[4]
    max_iterations = args[5]
    epsilon = args[6]
    gamma = args[7]
    learning_rate = args[8]

    model = RLModel(mode, weight_out, returns_out, int(episodes), int(max_iterations),
                    float(epsilon), float(gamma), float(learning_rate))
    model.outputAll()


if __name__ == "__main__":
    main(sys.argv)
