import numpy as np
import matplotlib.pylab as plt

from math import nan
from matplotlibrc import *
rcParams['figure.figsize'] = (7.0, 3.5)

## close all previously opened figure
plt.close('all')

##############################################
## SEQUENTIAL: SCALING w/ POINTS
##############################################
cells_scale_points = [[2.098, 1.92, 1.99],
                        [4.26901, 4.255, 4.28],
                        [22.251, 22.373, 22.604],
                        [45.977, 45.549, 45.771],
                        [nan, nan, nan],
                        [nan, nan, nan]]
cells_ave = np.mean(cells_scale_points, axis=1)
cells_std = np.std(cells_scale_points, axis=1)
print(cells_ave)
print(cells_std)
print(' ')

points_scale_points = [[0.002001, 0.0010029, 0.001],
                [0.0029995, 0.0020014, 0.002],
                [0.0130001, 0.0120012, 0.013014],
                [0.0250003, 0.0259928, 0.0249989],
                [0.718006, 0.803001, 0.863999],
                [7.32001, 7.28899, 7.62]]
points_ave = np.mean(points_scale_points, axis=1)
points_std = np.std(points_scale_points, axis=1)
print(points_ave)
print(points_std)
print(' ')

num_points = [5e2, 1e3, 5e3, 1e4, 1e5, 1e6]

fig = plt.figure(frameon=True)
ax = plt.gca()
plt.errorbar(num_points, cells_ave, yerr=cells_std, label='Cells looped')
plt.errorbar(num_points, points_ave, yerr=points_std, label='Points looped')
plt.xscale('log')
plt.grid(True)
plt.legend(loc='upper left')
ax.text(0.975, 0.9, '1e3 cells (a)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=20)
plt.xlabel('Number of points')
plt.ylabel('Time [s]')
plt.savefig('plot_seq_scale_points.png')
plt.close()

##############################################
## SEQUENTIAL: SCALING w/ CELLS
##############################################
cells_scale_cells = [[0.238997, 0.227996, 0.239015],
                        [5.544, 5.64901, 5.74],
                        [22.188, 22.373, 22.528],
                        [198.978, 198.991, 203.433],
                        [nan, nan, nan],
                        [nan, nan, nan]]
cells_ave = np.mean(cells_scale_cells, axis=1)
cells_std = np.std(cells_scale_cells, axis=1)
print(cells_ave)
print(cells_std)
print(' ')

points_scale_cells = [[0.0010007, 0.0019995, 0.0020089],
                        [0.0059989, 0.0059997, 0.0060016],
                        [0.0130082, 0.0119984, 0.011992],
                        [0.0369998, 0.0450009, 0.0359905],
                        [0.119001, 0.117998, 0.142001],
                        [0.246999, 0.240021, 0.238982]]
points_ave = np.mean(points_scale_cells, axis=1)
points_std = np.std(points_scale_cells, axis=1)
print(points_ave)
print(points_std)
print(' ')

num_cells = [1e2, 5e2, 1e3, 3e3, 1e4, 5e4]

fig = plt.figure(frameon=True)
ax = plt.gca()
plt.errorbar(num_cells, cells_ave, yerr=cells_std, label='Cells looped')
plt.errorbar(num_cells, points_ave, yerr=points_std, label='Points looped')
plt.xscale('log')
plt.grid(True)
plt.legend(loc='upper left')
ax.text(0.975, 0.9, '5e3 points (b)',
        verticalalignment='center', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=20)
plt.xlabel('Number of cells (per direction)')
plt.ylabel('Time [s]')
plt.savefig('plot_seq_scale_cells.png')
plt.close()

##############################################
## CPU SPEEDUP PLOT
##############################################
seq_times_1e6 = [9.96658, 9.87488, 9.88689]

CPU_times_1e6 = [[9.74617, 9.83632, 9.99506],
                [4.94956, 4.88838, 4.89819],
                [3.27199, 3.27933, 3.27218],
                [2.53925, 2.54011, 2.51317],
                [1.97465, 1.97530, 2.02599],
                [1.66016, 1.67281, 1.69535],
                [1.48751, 1.58038, 1.49044],
                [1.38791, 1.46906, 1.45669]]
num_procs = [1, 2, 3, 4, 5, 6, 7, 8]

speedup_cpu_seq = [[(tmp_seq/cpu) for cpu, tmp_seq in zip(cpu_array, seq_times_1e6)] for cpu_array in CPU_times_1e6]
speedup_cpu_seq_ave = np.mean(speedup_cpu_seq, axis=1)
speedup_cpu_seq_std = np.std(speedup_cpu_seq, axis=1)

## comparing sequential and CPU
fig = plt.figure(frameon=True)
plt.plot(num_procs, num_procs, 'k--', label='linear speedup')
plt.errorbar(num_procs, speedup_cpu_seq_ave, yerr=speedup_cpu_seq_std, label='CPU parallel')
plt.grid(True)
plt.legend(loc='upper left')
plt.xlabel('Number of processors')
plt.ylabel('Speedup')
plt.savefig('plot_speedup.png')
plt.close()

np.set_printoptions(suppress=True)

##############################################
## CPU/GPU/SEQ SCALING PLOT
##############################################
GPU_times_scale = [[0.126336,  0.118624, 0.110122],
                        [0.133186, 0.119981, 0.104505],
                        [0.134608, 0.123067, 0.12035],
                        [0.276124, 0.292831, 0.261159],
                        [2.64478,  2.64655,  2.63257]]
GPU_ave = np.mean(GPU_times_scale, axis=1)
GPU_std = np.std(GPU_times_scale, axis=1)

CPU_times_scale = [[0.0029649,  0.0020795, 0.0030308],
                        [0.0153262, 0.0168823, 0.0156866],
                        [0.145763,  0.149454,  0.148357],
                        [1.42271,   1.48265,   1.43344],
                        [14.4423,   14.4611,   14.4998]]
CPU_ave = np.mean(CPU_times_scale, axis=1)
CPU_std = np.std(CPU_times_scale, axis=1)

seq_times_scale = [[0.0135734, 0.0122912, 0.0113504],
                        [0.103968,  0.120809,  0.103222],
                        [1.01058,   1.00165,   0.990662],
                        [9.89547,   9.94896,   9.95081],
                        [98.1794,   98.1469,   98.1365]]
seq_ave = np.mean(seq_times_scale, axis=1)
seq_std = np.std(seq_times_scale, axis=1)

num_points = [1000, 10000, 100000, 1000000, 10000000]

fig = plt.figure(frameon=True)
ax = plt.gca()
plt.errorbar(num_points, GPU_ave, yerr=GPU_std, label='GPU parallelised')
plt.errorbar(num_points, CPU_ave, yerr=CPU_std, label='CPU parallelised')
plt.errorbar(num_points, seq_ave, yerr=seq_std, label='Sequential')
plt.xscale('log')
plt.grid(True)
plt.legend(loc='upper left')
plt.xlabel('Number of points')
plt.ylabel('Time [s]')
plt.savefig('plot_time.png')
plt.close()
