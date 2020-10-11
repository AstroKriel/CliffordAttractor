## load modules
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
from math import cos, sin

## close all previously opened figure
plt.close('all')

x_dim = 6.0
y_dim = 5

## create figure
fig = plt.figure(frameon=False)
fig.set_size_inches(x_dim, y_dim)
ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0], )
ax.set_axis_off()
fig.add_axes(ax)
## for each method
for str_method in ['cpu', 'gpu']:
    ## get list of all density files
    filepaths_density_data = sorted([filename for filename in os.listdir('.') if 
        filename.startswith("data_density") and filename.__contains__(str_method) and filename.endswith(".txt")])
    print('There are a total of ' + str(len(filepaths_density_data)) + ' ' + str_method + ' data files.')
    ## load and plot density data
    for filepath in filepaths_density_data:
        ## load data
        file_num = filepath.split('_density_')[1].split('.txt')[0]
        print('Loading density data set: ' + file_num)
        with open(filepath, 'r') as f:
            A = [[int(num) for num in line.split(', ')] for line in f]
        ## plot and save image with desired aspect ratio 
        plt.imshow(A, vmin=min(min(A)), vmax=0.95*max(max(A)), origin='lower', extent=[0, x_dim, 0, y_dim], cmap='viridis')
        plt.savefig('density_'+str(file_num)+'.png', dpi=300)
    print(' ')

## close figure
plt.close()
