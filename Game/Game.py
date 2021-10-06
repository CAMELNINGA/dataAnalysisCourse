import numpy as np
from keras.layers import Dense, Dropout, RNN, LSTM
from tensorflow.python.keras import Sequential

model = Sequential()
model.add(Dense(10, activation="sigmoid", input_shape=(2,)))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(1, activation='linear'))
model.compile(loss="mean_squared_error", optimizer="adam")

x_train = list(map(lambda s: list((map(lambda r: float(r), s.split(";")))), open("xdata.csv").readlines()))
c = np.array(x_train)
y_train = list(map(lambda s: float(s), open("ydata05.csv").readlines()))
f = np.array(y_train)
print(c)
print(f)
model.fit(c, f, epochs=10000)
