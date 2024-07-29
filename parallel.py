# import multiprocess as mp
import multiprocessing
import numpy as np

def my_function(x):
    return np.sin(x) * np.cos(x) + np.tan(x)

if __name__ == '__main__':
    with multiprocessing.Pool() as pool:
        results = pool.map(my_function, range(10000000))
    # print(results)