#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import math

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

# Make data.
X = np.arange(0, 1482, 1)
Y = np.arange(0, 3674, 1)

file = np.load('/home/lb/Downloads/argoverse-api/map_files/MIA_10316_driveable_area_mat_2019_05_28.npy')

for i in range(0,file.shape[0]):
    for j in range(0,file.shape[1]):
        if math.isnan(file[i][j]):
            file[i][j]=0;
            
X, Y = np.meshgrid(X, Y)

#surf = ax.plot_surface(X, Y, file, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.plot_wireframe(X, Y, file, rstride=200, cstride=200)

# Customize the z axis.
ax.set_zlim(0,5)  # z轴的取值范围
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
