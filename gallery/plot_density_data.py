## load modules
import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

def plotDensity(data_folder, filepath):
    ## dimensions of the figure
    x_dim = 6.0
    y_dim = 5
    ## create figure
    fig = plt.figure(frameon=False)
    fig.set_size_inches(x_dim, y_dim)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0], )
    ax.set_axis_off()
    fig.add_axes(ax)
    ## load data
    file_num = filepath.split('_density_')[1].split('.txt')[0]
    print('\t> Loading density data set: ' + file_num)
    with open((data_folder + '\\' + filepath), 'r') as f:
        A = [[int(num) for num in line.split(', ')] for line in f]
    ## plot and save image with desired aspect ratio 
    plt.imshow(A, vmin=min(min(A)), vmax=0.95*max(max(A)), origin='lower', extent=[0, x_dim, 0, y_dim], cmap='viridis')
    plt.savefig('.\images\density_'+str(file_num)+'.png', dpi=300)
    ## close figure
    plt.close()
    return 1

if __name__ == '__main__':
    ## close all previously opened figure
    plt.close('all')
    ## folder where the data is stored
    data_folder = '.\data\.'
    ## for each method
    for str_method in ['cpu', 'gpu', 'seq']:
        ## get list of all density files
        filepaths_density_data = sorted([filename for filename in os.listdir(data_folder) if 
            filename.startswith("data_density") and filename.__contains__(str_method) and filename.endswith(".txt")])
        print('There are a total of ' + str(len(filepaths_density_data)) + ' ' + str_method + ' data files.')
        ## load and plot density data in parallel
        pool = mp.Pool(processes=8)
        results = [pool.apply_async(plotDensity, args=(data_folder, filepath,)) for filepath in filepaths_density_data]
        results = [p.get() for p in results] ## need to extract info for parallel process to run properly
        print(' ')
    ## animate plots
    os.system('ffmpeg -start_number 0 -i .\images\density_%i*.png' + 
                ' -vb 40M -framerate 40 -vf scale=1440:-1 -vcodec mpeg4 ani_clifford_attractor.mp4')

