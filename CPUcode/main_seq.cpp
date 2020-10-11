#include <string>
#include <math.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <iomanip>
#include <sstream>

#include <chrono>  // for high_resolution_clock
#include <io.h>

// compile
// g++ .\main_seq.cpp -o program_seq

struct Pair { double min; double max; };  
struct Pair getMinMax(double* arr, int n) {
    struct Pair minmax;
    if (arr[0] > arr[1]) {
        minmax.max = arr[0];
        minmax.min = arr[1];
    } else {
        minmax.max = arr[1];
        minmax.min = arr[0];
    }
    for(int i = 2; i < n; i++) {
        if (arr[i] > minmax.max) { minmax.max = arr[i]; }
        else if (arr[i] < minmax.min) { minmax.min = arr[i]; }
    }
    return minmax;
}

void calcDensity_seq(double x_min, double x_max, double y_min, double y_max, 
                    int N, double* x, double* y, int M, int* A) {
    int tmp_row, tmp_col;
    double tmp_x, tmp_y, cell_width, cell_height;
    for (int point_id = 0; point_id < N; point_id++) {
        tmp_row = -1;
        tmp_col = -1;
        // save point coordinate
        tmp_x = x[point_id];
        tmp_y = y[point_id];
        // save width of cell
        cell_width = (x_max - x_min)/(double)M;
        cell_height = (y_max - y_min)/(double)M;
        // find the cell in which the point falls
        for (int i = 1; i < M; i++) {
            // check if point falls in cell's x-range
            if ((x_min+(double)i*cell_width <= tmp_x) && (tmp_x < x_min+(double)(i+1)*cell_width)) { tmp_col = i; }
            // check if point falls in cell's y-range
            if ((y_min+(double)i*cell_height <= tmp_y) && (tmp_y < y_min+(double)(i+1)*cell_height)) { tmp_row = i; }
        }
        // atomically increment cell count
        if (tmp_col > 0 && tmp_row > 0) { 
            A[tmp_row*M + tmp_col]++; 
        }
    }
}

void calcCliffordPoints(double a, double b, double c, double d, double *x, double *y, int N) {
    for (int i = 1; i < N; i++) {
        x[i] = sin(a * y[i-1]) + c*cos(a * x[i-1]);
        y[i] = sin(b * x[i-1]) + d*cos(b * y[i-1]);
    }
}

int main() {
    std::cout << "Started program.\n";
    std::cout << "\n";

    //======================================================
    // ALLOCATE SPACE FOR VARIABLES ON HOST
    //======================================================
    // define number of clifford points
    const int NUM_POINTS = 5e6;
    // allocate space on host for vectors 
    std::cout << "Allocating space on the host (CPU) for:\n";
    std::cout << "\t> " << NUM_POINTS << " clifford points.\n";
    // size of each vector
    const unsigned int bytes_points = NUM_POINTS * sizeof(double);
    double* points_x = (double*)malloc(bytes_points);
    double* points_y = (double*)malloc(bytes_points);
    // initialise the points
    points_x[0] = 0.0;
    points_y[0] = 0.0;
    for (int i = 1; i < NUM_POINTS; i++) {
        points_x[i] = 1.0;
        points_y[i] = 1.0;
    }
    // allocate space for density matrix (square matrix)
    const int SIZE_MATRIX = 1e3;
    std::cout << "\t> " << "Density matrix (" << SIZE_MATRIX << " x " << SIZE_MATRIX << ").\n";
    // size of the matrix
    const unsigned int bytes_mat = SIZE_MATRIX * SIZE_MATRIX * sizeof(int);
    int* seq_mat = (int*)malloc(bytes_mat);
    std::cout << "\n";

    //======================================================
    // PREPARE PARAMETER NAME
    //======================================================
    // define the parameter space
    double a = 1.5, b=-1.8, c=1.6, d=0.9;
    struct Pair x_range, y_range;
    std::stringstream stream_a, stream_b, stream_c, stream_d;
    stream_a << std::fixed << std::setprecision(3) << a;
    stream_b << std::fixed << std::setprecision(3) << b;
    stream_c << std::fixed << std::setprecision(3) << c;
    stream_d << std::fixed << std::setprecision(3) << d;
    // create parameter set name
    std::cout << "Analysing parameter set: " << "a=" << a << ", b=" << b << ", c=" << c << ", d=" << d << ".\n";
    std::string param_set_string = "a="+stream_a.str() + "_b="+stream_b.str() + "_c="+stream_c.str() + "_d="+stream_d.str();
    std::string param_set_string_gpu = "gpu_a="+stream_a.str() + "_b="+stream_b.str() + "_c="+stream_c.str() + "_d="+stream_d.str();
    std::string param_set_string_cpu = "cpu_a="+stream_a.str() + "_b="+stream_b.str() + "_c="+stream_c.str() + "_d="+stream_d.str();
    // reset string streams
    stream_a.str(std::string());
    stream_b.str(std::string());
    stream_c.str(std::string());
    stream_d.str(std::string());
    std::cout << "\n";

    //======================================================
    // CALCULATE CLIFFORD POINTS
    //======================================================
    std::cout << "Calculating clifford points...\n";
    calcCliffordPoints(a, b, c, d, points_x, points_y, NUM_POINTS);
    // display the first few points
    std::cout << "Checking subset of points...\n";
    for (int i = 0; i < 20; i++) {
        std::cout << "\t" << i << ": \t" << points_x[i] << ", " << points_y[i] << "\n";
    }
    std::cout << "\n";

    //======================================================
    // CALCULATE DENSITY MAP
    //======================================================
    // initialise the density maps
    for (int i = 0; i < SIZE_MATRIX; i++) {
        for (int j = 0; j < SIZE_MATRIX; j++) {
            seq_mat[i*SIZE_MATRIX + j] = (int)0;
        }
    }
    // find the bounds of the clifford attractor points
    x_range = getMinMax(points_x, NUM_POINTS);
    y_range = getMinMax(points_y, NUM_POINTS);

    // calculate the density map on the CPU sequentially
    std::cout << "Calculating density map sequentially...\n";
    auto seq_start = std::chrono::high_resolution_clock::now();
    calcDensity_seq(x_range.min, x_range.max, y_range.min, y_range.max, 
                        NUM_POINTS, points_x, points_y, SIZE_MATRIX, seq_mat);
    auto seq_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> seq_elapsed = seq_end - seq_start;
    std::cout << "Sequential computation elapsed: " << seq_elapsed.count() << " seconds\n";
    std::cout << "\n";

    //======================================================
    // RELEASE HOST (CPU) DATA
    //======================================================
    free(points_x);
    free(points_y);
    free(seq_mat);

    // end of program
    std::cout << "Finished program.\n";
    return 0;
}
