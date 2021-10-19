import matplotlib.pyplot as plt

# https://www.wsj.com/market-data/quotes/fx/EURUSD/historical-prices
# https://www.wsj.com/market-data/quotes/fx/EURRUB/historical-prices
# Date, Open, High, Low, Close
with open("HistoricalPrices2.csv") as csv_:
    # поиск параметров регрессии
    y = list(reversed(list(map(lambda s: float(s.split(", ")[4]), csv_.readlines()))))
    N = len(y)
    x = [i for i in range(N)]
    x_ = sum(x)
    y_ = sum(y)
    xy_ = sum(map(lambda s: s[0] * s[1], [[x[i], y[i]] for i in range(N)]))
    x2_ = sum([i * i for i in x])
    b1 = (N * xy_ - x_ * y_) / (N * x2_ - x_ * x_)
    b0 = (y_ - b1 * x_) / N
    print(f"Regression function: C = {b0} {'+' if b1 > 0 else '-'} {abs(b1)} * t + e")

    # коэфициент детерминации
    y_ = y_ / N
    sse = sum(map(lambda s: (s[0] - s[1]) * (s[0] - s[1]), [[y[i], b0 + b1 * x[i]] for i in range(N)]))
    sst = sum(map(lambda s: (s[0] - s[1]) * (s[0] - s[1]), [[y[i], y_] for i in range(N)]))
    r2 = 1 - sse / sst
    r2_adj = 1 - (1 - r2) * (N - 1) / (N - 2)
    print(f"R^2 = {r2}")
    print(f"Adjusted R^2 = {r2_adj}")
    # графики

    plt.plot(x, y, 'ro')
    plt.plot(x, [b0 + b1 * x[i] for i in range(N)])
    plt.show()
