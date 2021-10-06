import math


class Perceptron2:
    rate = 0.5

    def sigmoid(self, x, der=False):
        if not der:
            return 1 / (1 + math.exp(-x))
        else:
            return x * (1 - x)

    def __init__(self, input_dim, hidden_layer_dim, output_layer_dim):
        self.inp = input_dim
        self.hid = hidden_layer_dim
        self.out = output_layer_dim
        self.activation = self.sigmoid
        self.weights = list()
        self.weights.append([[0] * self.hid for i in range(self.inp)])
        self.weights.append([[0] * self.out for i in range(self.hid)])

    def predict(self, inputs, info=False):
        sums = list()
        outputs = list()
        for i in range(self.hid):
            summary = 0
            for j in range(self.inp):
                summary += self.weights[0][j][i] * inputs[j]
            sums.append(summary)
            outputs.append(self.activation(summary))

        sums2 = list()
        outputs2 = list()
        for i in range(self.out):
            summary = 0
            for j in range(self.hid):
                summary += self.weights[1][j][i] * outputs[j]
            sums2.append(summary)
            outputs2.append(self.activation(summary))

        if not info:
            return outputs2

        else:
            return outputs2, sums, sums2, outputs, outputs2

    def study(self, inputs_array, outputs_array, epochs=100):
        for epoch in range(epochs):
            sum_error = 0
            count = 0
            for i in range(len(inputs_array)):
                inps = inputs_array[i]
                outs = outputs_array[i]
                prediction, sum1, sum2, out1, out2 = self.predict(inps, info=True)
                error = 0
                for j in range(len(prediction)):
                    error += 1 / 2 * ((prediction[j] - outs[j]) ** 2)
                sum_error += error
                count += 1
                for j in range(self.hid):
                    for k in range(self.inp):
                        total = 0
                        for l in range(self.out):
                            total += -(outs[l] - out2[l]) * self.activation(out2[l], der=True) * \
                                     self.weights[1][j][l]
                        out_h = self.activation(out1[j], der=True)
                        net_h = inps[k]
                        self.weights[0][k][j] -= self.rate * total * out_h * net_h

                for j in range(self.out):
                    for k in range(self.hid):
                        self.weights[1][k][j] += self.rate * (outs[j] - out2[j]) * self.activation(out2[j],
                                                                                                         der=True) * \
                                                 out1[k]
            print(f"epoch - {epoch}/{epochs}. Avg error - {sum_error/count}")


if __name__ == '__main__':
    y = list(map(lambda string: float(string.split(",")[7]), open('lines.csv').readlines()))
    x = [i + 1 for i in range(len(y))]
    inps = list()
    outs = list()
    for i in range(len(y) - 13):
        inps.append(y[i:i + 10])
        x_ = sum(x[i + 10:i + 13]) / 3
        y_ = sum(y[i + 10:i + 13]) / 3
        xy_ = 0
        for j in range(3):
            try:
                xy_ += x[i + 10 + j] * y[i + 10 + j]
            except IndexError:
                print(i, j)
        xy_ /= 3
        x2_ = sum(map(lambda s: s * s, x[i + 10:i + 13])) / 3
        a = (xy_ - x_ * y_) / (x2_ - x_ * x_)
        b = y_ - a * x_
        outs.append([1] if a >= 0 else [-1])

    model = Perceptron2(10, 5, 1)
    model.study(inps, outs, epochs=100)
