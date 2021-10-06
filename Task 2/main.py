import random
import time
import matplotlib.pyplot as plt


class SinglePerceptron:
    def __init__(self, inputs_num: int):
        self.input_len = inputs_num
        self.weights = [random.random() for i in range(inputs_num)]

    def predict(self, input_array: list):
        if len(input_array) != self.input_len:
            raise IOError()
        sum_ = 0
        for i in range(self.input_len):
            sum_ += self.weights[i] * input_array[i]

        # activation function

        y = 1 if sum_ > 0 else -1

        return y

    def fit(self, epochs, array_of_inputs, array_of_outputs):
        errs = -1
        tries = 0
        for k in range(epochs):
            errs = 0
            for i in range(len(array_of_inputs)):
                row = array_of_inputs[i]  # x, y
                prediction = self.predict(row)
                if prediction * array_of_outputs[i] < 0:
                    errs += 1
                    for j in range(len(self.weights)):
                        self.weights[j] += 0.01 * row[j] * array_of_outputs[i]
            if k % 10 == 0:
                print(100 - (errs / len(array_of_inputs) * 100), end="%\n")


if __name__ == '__main__':
    parsed = list(map(lambda s: list(map(lambda x: int(x), s.split(";"))), open('vk_perc.csv').readlines()))
    inputs = list(map(lambda s: s[1:4], parsed))
    print(inputs)
    outputs = list(map(lambda s: 1 if s[0] == 4 else -1, parsed))
    print(outputs)
    model = SinglePerceptron(3)
    model.fit(100, inputs, outputs)
#    for i in range(500, len(x)):
