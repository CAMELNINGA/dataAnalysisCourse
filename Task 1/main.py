import random
import time
import matplotlib.pyplot as plt

# working version
class SinglePerceptron:
    def __init__(self, inputs_num: int, delta: float):
        self.input_len = inputs_num
        self.weights = [random.random(), 1]
        self.weights.append(-delta)

    def predict(self, input_array: list):
        if len(input_array) != self.input_len:
            raise IOError()
        sum_ = self.weights[self.input_len]  # sum = bias at start
        for i in range(self.input_len):
            sum_ += self.weights[i] * input_array[i]

        # activation function

        y = 1 if sum_ > 0 else -1

        return y

    def fit(self, array_of_inputs, array_of_outputs):
        errs = -1
        tries = 0
        min_errs = len(array_of_inputs)
        while errs != 0:
            errs = 0
            for i in range(len(array_of_inputs)):
                row = array_of_inputs[i]  # x, y
                prediction = self.predict(row)
                if prediction * array_of_outputs[i] < 0:
                    errs += 1
                    self.weights[0] += 0.001 * row[0] * array_of_outputs[i]
            temp_ = min_errs
            min_errs = min(errs, min_errs)
            if temp_ != min_errs:
                print(min_errs)
                print(*self.weights)
                plt.plot(x, y)
                plt.plot([0, len(x)], [a * 0 + b, a * len(x) + b])
                plt.plot([0, len(array_of_inputs)], [-self.weights[0] / self.weights[1] * 0 - self.weights[2],
                                                     -self.weights[0] / self.weights[1] * len(array_of_inputs) -
                                                     self.weights[2]])
                plt.show()


if __name__ == '__main__':
    y = list(map(lambda string: float(string.split(",")[7]), open('lines.csv').readlines()))
    x = [i + 1 for i in range(len(y))]
    x_ = sum(x) / len(x)
    y_ = sum(y) / len(y)
    xy_ = 0
    for i in range(len(y)):
        xy_ += x[i] * y[i]
    xy_ /= len(y)
    x2_ = sum(map(lambda s: s * s, x)) / len(x)
    a = (xy_ - x_ * y_) / (x2_ - x_ * x_)
    b = y_ - a * x_
    print(f"a={a}, b={b}")
    answers = list()
    for i in range(len(x)):
        answers.append(1 if a * x[i] + b <= y[i] else -1)
    inputs = [[i, y[i]] for i in range(len(y))]
    print(x)
    print(y)
    plt.plot(x, y)
    plt.plot([0, len(x)], [a * 0 + b, a * len(x) + b])
    plt.show()
    model = SinglePerceptron(2, 3134.8901260195853)
    model.fit(inputs, answers)
#    for i in range(500, len(x)):
