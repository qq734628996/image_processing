from PIL import Image
import os
import cv2
from skimage import io, filters
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(19260817)


def get_images():
    '''
    读取所有图像
    '''
    images = []
    for i in range(6):
        path = os.path.join('pic', 'A{}.png'.format(i))
        image = io.imread(path, as_gray=True)
        images.append(image)
    return images


def get_edges(image):
    '''
    提取边缘，采用sobel算子
    '''
    edges = filters.sobel(image)
    return edges


def get_points(edges):
    '''
    先对图像进行二值化，再抽离出所有点。
    '''
    mean = np.mean(edges[edges > 0])
    edges = edges > mean
    res = []
    row, col = edges.shape
    for i in range(row):
        for j in range(col):
            if edges[i][j]:
                res.append((i, j))
    return np.array(res)


def sample(points, size=100):
    '''
    对抽离出来的点进行随机采样，数目为size个。
    '''
    idx = np.random.choice(np.arange(len(points)), size=size)
    return points[idx]


def save_points(points, shape, name):
    '''
    将点转化为图像并保存。
    '''
    for i, r in enumerate(points):
        image = np.zeros(shape=shape, dtype=np.uint8)
        for p in r:
            image[p[0], p[1]] = 255
        image = 255-image
        path = os.path.join('pic', '{}{}.png'.format(name, i))
        Image.fromarray(image).save(path)


def k_bin(A, k=12, bins=3):
    '''
    计算k-bin统计图，默认方位12等分，最大距离1000(log1000=3)
    '''
    n = len(A)
    k_bin = np.zeros(shape=(n, k, bins), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            theta = np.arctan2(A[j][1]-A[i][1], A[j][0]-A[i][0])
            dis = np.hypot(A[j][1]-A[i][1], A[j][0]-A[i][0])
            x = int(np.floor((theta/np.pi+1)*6-1e-6))
            y = int(np.floor(np.log10(dis+1)))
            k_bin[i][x][y] += 1
    return k_bin


def get_cost(A, B, k=12, bins=3):
    '''
    计算花费矩阵
    '''
    k_bin_A = k_bin(A, k, bins)
    k_bin_B = k_bin(B, k, bins)
    nA = len(A)
    nB = len(B)
    cost = np.zeros(shape=(nA, nB))
    for i in range(nA):
        for j in range(nB):
            m, n = k_bin_A[i], k_bin_B[j]
            t = m+n > 0
            m, n = m[t], n[t]
            cost[i][j] = np.sum((m-n)**2/(m+n+1e-6))/2
    assert(np.sum(np.isnan(cost)) == 0)
    return cost


def KM(g):
    '''
    KM算法，计算带权二分图的最小费用。
    Input:
        np.array([
            [3, 4, 6, 4, 9],
            [6, 4, 5, 3, 8],
            [7, 5, 3, 4, 2],
            [6, 3, 2, 2, 5],
            [8, 4, 5, 4, 7],
        ], dtype=np.float64)
    Output:
        29
    '''
    nx, ny = g.shape
    linker = np.full(ny, fill_value=-1, dtype=np.int32)
    lx = np.zeros(nx, dtype=np.float64)
    ly = np.zeros(ny, dtype=np.float64)
    slack = np.zeros(ny, dtype=np.float64)
    vis_x = np.zeros(nx, dtype=np.bool)
    vis_y = np.zeros(ny, dtype=np.bool)

    def dfs(x):
        nonlocal g, ny, linker, lx, ly, slack, vis_x, vis_y
        vis_x[x] = True
        for y in range(ny):
            if vis_y[y]:
                continue
            t = lx[x]+ly[y]-g[x][y]
            if (np.abs(t) < 1e-6):
                vis_y[y] = True
                if linker[y] == -1 or dfs(linker[y]):
                    linker[y] = x
                    return True
            elif slack[y] > t:
                slack[y] = t
        return False

    lx = np.max(g, axis=1)
    for x in range(nx):
        slack[:] = np.inf
        while True:
            vis_x[:], vis_y[:] = False, False
            if dfs(x):
                break
            d = np.min(slack[vis_y == False])
            lx[vis_x] -= d
            ly[vis_y] += d
            slack[vis_y == False] -= d
    t = linker != -1
    res = np.sum(g[linker[t], t])
    return res


def shape_context(A, B):
    '''
    计算形状上下文的值，这里只计算到带权二分图的费用，
    实际论文中后面还有几个步骤没有实现。
    '''
    cost = get_cost(A, B)
    return KM(cost)


'''
采样点数 结果
50: 2053
100: 8020
200: 32142

100个采样点下，不同结果
1: 8020
2: 8091
3: 8104
4: 7990
5: 8326
'''


def main():
    images = get_images()
    shape = images[0].shape
    edges = [get_edges(i) for i in images]
    points = [get_points(i) for i in edges]
    points_sample = [sample(i) for i in points]
    save_points(points, shape, 'A_points')
    save_points(points_sample, shape, 'A_points_sample')
    print(shape_context(points_sample[0], points_sample[5]))


if __name__ == '__main__':
    main()
