import matplotlib.pyplot as plt

# Date, Open, High, Low, Close
with open("HistoricalPrices2.csv") as csv_:
    # поиск параметров регрессии
    lines = csv_.readlines()
    y = list(map(lambda s: float(s.split(", ")[4]), lines))
    N = len(y)
    x = list(map(lambda s: float(s.split(", ")[1]), lines))
    x_ = sum(x)
    y_ = sum(y)
    xy_ = sum(map(lambda s: s[0] * s[1], [[x[i], y[i]] for i in range(N)]))
    x2_ = sum([i * i for i in x])
    b1 = (N * xy_ - x_ * y_) / (N * x2_ - N * x_ * x_)
    b0 = (y_ - b1 * x_) / N
    print(f"Regression function: C = {b0} {'+' if b1 > 0 else '-'} {abs(b1)} * O + e")

    # коэфициент детерминации
    y_ = y_/N
    sse = sum(map(lambda s: (s[0] - s[1]) * (s[0] - s[1]), [[y[i], b0 + b1 * x[i]] for i in range(N)]))
    sst = sum(map(lambda s: (s[0] - s[1]) * (s[0] - s[1]), [[y[i], y_] for i in range(N)]))
    r2 = 1 - sse / sst
    print(f"R^2 = {r2}")

    # графики

    plt.plot(x, y, 'ro')
    plt.plot(x, [b0 + b1 * x[i] for i in range(N)])
    plt.show()
