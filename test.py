import sys
sys.path.append('sequential-line-search/build/python/')

import pySequentialLineSearch as sls

import numpy as np
np.set_printoptions(suppress=True)

target = np.float64([0.1, 0.2, 0.3, 0.4, 0.5]);
print(target)

dims = len(target)
sls.init(dims)

for it in range(10):
    a = np.float64(sls.getParametersFromSlider(0.0))
    b = np.float64(sls.getParametersFromSlider(1.0))
    print(a, b)
    d = b-a

    x = np.clip(np.dot(d, target-a) / np.dot(d, d), 0, 1)
    print(x)
    sls.proceedOptimization(x)

print('best:', np.array(sls.getXmax()))
