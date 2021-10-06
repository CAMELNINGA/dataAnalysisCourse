import numpy as np
from keras.layers import Dense
from tensorflow.python.keras import Sequential

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
        answers.append(1 if a * x[i] + b >= y[i] else 0)
    inputs = [[i, y[i]] for i in range(len(y))]
    model = Sequential()

    model.add(Dense(1, input_shape=(2,), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    c = np.array(inputs)
    f = np.array(answers)

    model.fit(c, f, epochs=100)
    print(model.get_weights())
    print(*inputs[500:])
    for i in range(500, len(inputs)):
        predicted = model.predict([inputs[i]])
        found = 1 if predicted[0][0] >= 0.5 else 0
        actual = answers[i]
        print(actual, found, predicted[0][0])
