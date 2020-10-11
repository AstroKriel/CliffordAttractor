#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string>
#include <math.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <iomanip>
#include <sstream>

namespace Wrapper {
	void device_wrapper(double x_min, double x_max, double y_min, double y_max, 
                        int NUM_POINTS, double* host_x, double* host_y, unsigned int bytes_points, 
                        int SIZE_MATRIX, int* host_mat, unsigned int bytes_mat);
}
