import numpy as np
import numpy.matlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import matplotlib

def randrange(n, vmin, vmax):
    return (vmax - vmin)*np.random.rand(n) + vmin

filter_num = 100
filter_size = 3
np.random.seed(47)
w = 2*np.random.rand(filter_num,filter_size)-1

#for x in np.nditer(w, op_flags=['readwrite']):
#    x[...] = x + 1 if x > 0 else x -1

matplotlib.rcParams.update({'font.size': 22})
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(0,10):
    temp_ = w**2
    temp_n_ = np.reshape(np.sum(temp_,axis=1),(filter_num,1))
    temp_n_ = np.sqrt(temp_n_)
    temp_ = np.matlib.repmat(temp_n_,1,filter_size)
    w_norm = np.divide(w,temp_)

    forces = filter_num*w_norm - np.matlib.repmat(np.sum(w_norm,axis=0),filter_num,1)
    projection = np.sum(np.multiply(forces,w_norm),axis=1).reshape(filter_num,1)
    regularization = forces - np.multiply( np.matlib.repmat(projection,1,filter_size),w_norm)
    norm_regularization = np.multiply(regularization,temp_)
    w = w - norm_regularization * 0.005

print w
ax.scatter(w[:, 0], w[:, 1], w[:, 2], c='r', marker='o', alpha=0.4, s=400,depthshade=True)
ax.scatter(0,0,0, c='b', marker='x', alpha=0.5, s=400,depthshade=True)

lim_val = 1.5
plt.xlim(-lim_val,lim_val)
plt.ylim(-lim_val,lim_val)
ax.set_zlim(-lim_val,lim_val)
ax.set_xticks(np.arange(-1,1+0.1,1))
ax.set_yticks(np.arange(-1,1+0.1,1))
ax.set_zticks(np.arange(-1,1+0.1,1))
plt.show()