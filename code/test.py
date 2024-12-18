import numpy as np
import matplotlib.pyplot as plt
from utils import test_example
from exp import discretize, get_discretization


n = 2000
fun_, _ = test_example(100)
samples = np.random.randn(n,2)
grid_x, grid_y = get_discretization( 2, 2, 0.05, 0.05 )
prob = discretize(fun_, grid_x, grid_y, unorm_log=True)
print(prob.shape)
Z = np.exp(fun_([grid_x,grid_y]))
Z /= np.sum(Z)

#plt.pcolormesh(grid_x, grid_y, np.exp(prob)/c, shading='auto', cmap='Reds')
plt.contourf(grid_x[:-1,:-1], grid_y[:-1,:-1], prob, levels=100)
#plt.contourf(grid_x, grid_y, Z, levels=100)

plt.colorbar()
plt.title('Target Distribution')
plt.tight_layout()
plt.show()
