from data_structure import Mat
import numpy as np
import matplotlib.pyplot as plt

x = np.random.uniform(0, 5, (100, ))
y = np.random.uniform(0, 5, (100, ))
label = np.zeros((100,))

index = 0
for _x,_y in zip(x,y):
    if _x - _y  > 0:
        label[index] = 1
    else:
        label[index] = 0
    index += 1


plt.scatter(x[label==1],y[label==1])
plt.scatter(x[label==0],y[label==0])
plt.show()