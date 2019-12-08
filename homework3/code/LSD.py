import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform, data, filters
import LinearRegression as LR
np.random.seed(19260817)


def scatter2image(X, y):
    plt.scatter(X, y, color='black', marker='.')
    plt.axis('off')
    path = os.path.join('img', 'data.png')
    plt.savefig(path)
    plt.close()
    image = io.imread(path, as_gray=True)
    image = 1 - image
    io.imsave(path, image)
    return image


# http://blog.itpub.net/31077337/viewspace-2213246/
def hough_line(img):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = img.shape
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))   # Dmax
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)
    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges
    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len)
            accumulator[rho, t_idx] += 1
    return accumulator, thetas, rhos


# https://www.cnblogs.com/denny402/p/5158707.html
def show_transform(accumulator, path):
    image = np.log(1+accumulator)
    image = transform.resize(image, (512, 512))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(os.path.join('img', path))
    plt.close()


def show_line(image, accumulator, thetas, rhos, threshold, path):
    io.imshow(image)
    row, col = image.shape
    for _, angle, dist in zip(*transform.hough_line_peaks(accumulator, thetas, rhos, threshold=threshold)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - col * np.cos(angle)) / np.sin(angle)
        plt.plot((0, col), (y0, y1), '-c')
    plt.axis((0, col, row, 0))
    path = os.path.join('img', path+str(threshold))
    plt.savefig(path)
    plt.close()


def test1():
    X, y = LR.make_data()
    image = scatter2image(X, y)
    accumulator, thetas, rhos = transform.hough_line(
        image)  # hough_line(image)
    show_transform(accumulator, 'hough_transform')
    show_line(image, accumulator, thetas, rhos, 50, 'hough_line')


def get_image():
    path = os.path.join('img', 'desk.png')
    image = io.imread(path, as_gray=True)
    edges = filters.sobel(image)
    io.imshow(edges)
    plt.savefig(os.path.join('img', 'desk_sobel.png'), dpi=300)
    plt.close()
    return edges


def image2scatter(image, threshold=0.1):
    row, col = image.shape
    X, y = [], []
    for i in range(row):
        for j in range(col):
            if image[i][j] > threshold:
                X.append([row-1-i])
                y.append(j)
    image = (image > threshold) * 1.0
    plt.imshow(image, plt.cm.gray)
    X = np.asarray(X).reshape((len(X), 1))
    y = np.asarray(y).reshape((len(y), 1))
    return X, y


def test2_1(image):
    X, y = image2scatter(image)
    models = [
        LR.LinearLeastSquare,
        LR.PolynomialLeastSquares,
        LR.RANSAC,
    ]
    for m in models:
        model = m()
        model.fit(X, y)
        X_test = np.linspace(X.min(), X.max())[:, np.newaxis]
        y_pred = model.predict(X_test)
        print(m.__name__, 'MSE:', model.score(X, y))
        plt.plot(X_test, y_pred, label=m.__name__)
    plt.legend(loc='lower right')
    plt.savefig(os.path.join('img', 'desk_LinearRegression.png'))

def test2_2(image):
    image = (image > 0.1) * 1.0
    accumulator, thetas, rhos = transform.hough_line(image)
    show_transform(accumulator, 'desk_hough_transform')
    show_line(image, accumulator, thetas, rhos, 250, 'desk_hough_line')


def test2():
    image = get_image()
    # test2_1(image)
    test2_2(image)


def main():
    # test1()
    test2()


if __name__ == '__main__':
    main()
