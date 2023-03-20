import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

def get_feed(y):
    feed_min = 0.01
    feed_range = 0.09

    feed = feed_min + (feed_range * y)
    return feed

def get_kill_range(y):
    kill_min = 0.045
    kill_range = 0.025
    feed_range = 0.09

    feed = get_feed(y)

    # kill_arc = (11 * kill_range / 16) - (((feed - (3 * feed_range / 5)) / (feed_range / 50)) ** 2)
    kill_arc = kill_min + (8 * kill_range / 32) + (feed / 3) - (((feed - (38 * feed_range / 100)) * 2.5) ** 2)
    kill_low = kill_arc - (1 / ((feed + 0.02) * 4000))
    kill_high = kill_low + ((2 / ((feed + 0.02) * 4000)) * 1)
    return feed, max(kill_low, kill_min), min(kill_high, kill_min + kill_range)

def main():
    
    img = mpimg.imread('reactioncurve.png')
    imgplot = plt.imshow(img, extent=[0.045, 0.07, 0.1015, 0.006], aspect="auto")

    y_low_points = np.array([0.022, 0.035, 0.05, 0.065, 0.08, 0.1]).reshape(-1, 1)
    x_low_points = [0.045, 0.0555, 0.0595, 0.060, 0.059, 0.0555]
    y_high_points = np.array([0.01, 0.024, 0.045, 0.06, 0.08, 0.1]).reshape(-1, 1)
    x_high_points = [0.0545, 0.063, 0.067, 0.067, 0.065, 0.0605]

    plt.scatter(x_low_points, y_low_points)
    plt.scatter(x_high_points, y_high_points)

    ys = []
    xs_low = []
    xs_high = []

    for y in np.linspace(0, 1, 20):
        feed = get_feed(y)
        ys.append(feed)

    ys = np.array(ys).reshape(-1, 1)

    poly = PolynomialFeatures(degree=3)
    y_low_points_ = poly.fit_transform(y_low_points)
    y_high_points_ = poly.fit_transform(y_high_points)
    ys_ = poly.fit_transform(ys)

    kill_low_lr = linear_model.LinearRegression()
    kill_low_lr.fit(y_low_points_, x_low_points)
    xs_low = kill_low_lr.predict(ys_)

    kill_high_lr = linear_model.LinearRegression()
    kill_high_lr.fit(y_high_points_, x_high_points)
    xs_high = kill_high_lr.predict(ys_)
    
    plt.scatter(xs_low, ys)
    plt.scatter(xs_high, ys)

    plt.show()

if __name__ == '__main__':
    main()