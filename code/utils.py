import numpy as np

scale = np.array([1,25]).reshape(1,2)
rand_0_1 = np.random.normal(0, 1, size=(10,2))
rand = np.random.normal(0, scale, size=(10,2))

print(rand)
print(scale*rand_0_1)