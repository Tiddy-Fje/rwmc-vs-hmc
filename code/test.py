import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 5., 100)

alpha = -2.
log_thinghy = np.log( 1 + np.exp(alpha*x) ) 


plt.plot(x, log_thinghy / (np.log(2)*np.exp(alpha*x)))
#plt.plot(x, np.log(2)+alpha*x)
#plt.plot(x, np.log(2)*np.exp(alpha*x))
#plt.show()
