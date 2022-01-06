{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import numpy as np\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Node():\n",
    "    def __init__(self, train_in, test_in, split_ind):\n",
    "        self.data = np.loadtxt(train_in, dtype=str, delimiter='\\t') # train_in\n",
    "        self.test_data = np.loadtxt(test_in, dtype=str, delimiter='\\t')# test_in\n",
    "        self.split_ind = int(split_ind)\n",
    "        self.values = [] # (2) values of given attribute, index 0 = a, index 1 = b\n",
    "        self.votes = [] # (2) values of votes (ex. demo/repub)\n",
    "        self.a = [] # list of votes from one of the split sides\n",
    "        self.b = []\n",
    "        self.a_vote = None\n",
    "        self.b_vote = None\n",
    "        self.train_predict = [] # prediction of labels\n",
    "        self.test_predict = []\n",
    "        self.train_error = 0 # error as float (wrong/total)\n",
    "        self.test_error = 0\n",
    "\n",
    "\n",
    "    def train(self): # maps self.values[0] to self.a_vote, and same for b\n",
    "        print(self.data[0])\n",
    "        for i in range(len(self.data)): # split data into 'a' and 'b' groups\n",
    "            if i == 0: continue\n",
    "            val = self.data[i][self.split_ind]\n",
    "            vote = self.data[i][-1]\n",
    "            if val in self.values:\n",
    "\n",
    "                if self.values.index(val) == 0:\n",
    "                    self.a.append(vote)\n",
    "                    if vote not in self.votes:\n",
    "                        self.votes.append(vote)\n",
    "                else:\n",
    "                    self.b.append(vote)\n",
    "                    if vote not in self.votes:\n",
    "                        self.votes.append(vote)\n",
    "            if len(self.values) == 0:\n",
    "                self.values.append(val)\n",
    "                self.a.append(vote)\n",
    "            else:\n",
    "                self.values.append(val)\n",
    "                self.b.append(self.data[i][-1])\n",
    "        self.a_vote = self.findMajority(self.a)\n",
    "        if self.a_vote == self.votes[0]: # second vote is just the remaining option, not another majority\n",
    "            self.b_vote = self.votes[1]\n",
    "        else:\n",
    "            self.b_vote = self.votes[0]\n",
    "        #self.b_vote = self.findMajority(self.b)\n",
    "\n",
    "    def findMajority(self, data): # returns the majority vote given a dataset\n",
    "        first = data.count(self.votes[0])\n",
    "        second = data.count(self.votes[1])\n",
    "        if first > second:\n",
    "            return self.votes[0]\n",
    "        else:\n",
    "            return self.votes[1] # if votes are equal, defaults to the second choice\n",
    "\n",
    "    def test(self, data, filename, predictions):\n",
    "        file = open(filename, 'w')\n",
    "        for i in range(len(data)):\n",
    "            if i == 0: continue\n",
    "            val = data[i][self.split_ind]\n",
    "            if val == self.values[0]:\n",
    "                vote = self.a_vote\n",
    "            else:\n",
    "                vote = self.b_vote\n",
    "            predictions.append(vote)\n",
    "            file.write(\"%s\\n\" %vote)\n",
    "        file.close()\n",
    "\n",
    "    def findError(self, data, predictions):\n",
    "        total = 0\n",
    "        wrong = 0\n",
    "        for i in range(len(predictions)):\n",
    "            if predictions[i] != data[i+1][-1]:\n",
    "                wrong += 1\n",
    "            total += 1\n",
    "        error = float(wrong)/total\n",
    "        return error\n",
    "\n",
    "    def outputErrors(self, filename):\n",
    "        file = open(filename, 'w')\n",
    "        file.write(\"error(train): %f\\n\" % self.train_error)\n",
    "        file.write(\"error(test): %f\\n\" % self.test_error)\n",
    "        file.close()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "def main():\n",
    "    train_in = sys.argv[1]\n",
    "    test_in = sys.argv[2]\n",
    "    split_ind = sys.argv[3]\n",
    "    train_out = sys.argv[4]\n",
    "    test_out = sys.argv[5]\n",
    "    metrics_out = sys.argv[6]\n",
    "\n",
    "    Stump = Node(train_in, test_in, split_ind)\n",
    "    Stump.train()\n",
    "    Stump.test(Stump.data, \"%s\" % train_out, Stump.train_predict)\n",
    "    Stump.test(Stump.test_data, \"%s\" % test_out, Stump.test_predict)\n",
    "\n",
    "    Stump.train_error = Stump.findError(Stump.data, Stump.train_predict)\n",
    "    Stump.test_error = Stump.findError(Stump.test_data, Stump.test_predict)\n",
    "    Stump.outputErrors(\"%s\" % metrics_out)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "ename": "IndexError",
     "evalue": "list index out of range",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mIndexError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-11-c7bc734e5e35>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0m__name__\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;34m'__main__'\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m     \u001b[0mmain\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m<ipython-input-10-d3a3e6b216ed>\u001b[0m in \u001b[0;36mmain\u001b[0;34m()\u001b[0m\n\u001b[1;32m      2\u001b[0m     \u001b[0mtrain_in\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0msys\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0margv\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m     \u001b[0mtest_in\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0msys\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0margv\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m2\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 4\u001b[0;31m     \u001b[0msplit_ind\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0msys\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0margv\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m3\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      5\u001b[0m     \u001b[0mtrain_out\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0msys\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0margv\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m4\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m     \u001b[0mtest_out\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0msys\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0margv\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m5\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mIndexError\u001b[0m: list index out of range"
     ]
    }
   ],
   "source": [
    "if __name__ == '__main__':\n",
    "    main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
