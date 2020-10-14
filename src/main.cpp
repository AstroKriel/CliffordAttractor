// #include <cuda.h>
// #include <io.h>
// #include "omp.h"
// #include <string>
// #include <math.h>
// #include <stdio.h>
// #include <stdexcept>

#include <chrono>   // for high_resolution_clock
#include <iostream> // for std::cout
#include <iomanip>  // for setting precision of output strings
#include <sstream>  // for converting numbers to strings streams
#include <fstream>  // for creating string streams

#include "calc_density_GPU.cuh"
#include "calc_density_CPU.h"
#include "calc_density_SEQ.h"

// compile program
// nvcc .\src\main.cpp .\src\calc_density_CPU.cpp .\src\calc_density_SEQ.cpp .\src\calc_density_GPU.cu  -I.\include\ -Xcompiler -openmp -o program

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

void calcCliffordPoints(double a, double b, double c, double d, double *x, double *y, int N) {
    for (int i = 1; i < N; i++) {
        x[i] = sin(a * y[i-1]) + c*cos(a * x[i-1]);
        y[i] = sin(b * x[i-1]) + d*cos(b * y[i-1]);
    }
}

int main() {
    std::cout << "Started program.\n";
    std::cout << "\n";

    // define number of clifford points
    const int NUM_POINTS = 1e7;
    // define the resolution of the grid domain (number of matrix rows/columns)
    const int SIZE_MATRIX = 1e3;
    // choose which implementations to check
    const bool run_GPU = false;
    const bool run_CPU = true;
    const bool run_SEQ = false;
    // choose if clifford points should be saved
    const bool save_points = false;
    // choose if density maps should be saved
    const bool save_density = false;

    //======================================================
    // ALLOCATE SPACE FOR VARIABLES ON HOST
    //======================================================
    // allocate space on host for vectors 
    std::cout << "Allocating space on the host (CPU) for:\n";
    std::cout << "\t> " << NUM_POINTS << " clifford points.\n";
    const unsigned int bytes_points = NUM_POINTS * sizeof(double);
    double* points_x = (double*)malloc(bytes_points);
    double* points_y = (double*)malloc(bytes_points);
    // initialise the points
    for (int i = 0; i < NUM_POINTS; i++) {
        points_x[i] = 0.0;
        points_y[i] = 0.0;
    }
    // allocate space for density matrix (square matrix)
    std::cout << "\t> " << "Density matrix (" << SIZE_MATRIX << " x " << SIZE_MATRIX << ").\n";
    const unsigned int bytes_mat = SIZE_MATRIX * SIZE_MATRIX * sizeof(int);
    int* cuda_host_mat = (int*)malloc(bytes_mat);
    int* openMP_mat = (int*)malloc(bytes_mat);
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
    std::cout << "Analysing parameter set: " << "a = " << a << ", b = " << b << ", c = " << c << ", d = " << d << ".\n";
    std::string param_set_string = "a="+stream_a.str() + "_b="+stream_b.str() + "_c="+stream_c.str() + "_d="+stream_d.str();
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
    for (int i = 0; i < 10; i++) { std::cout << "\t" << i << ": \t" << points_x[i] << ", " << points_y[i] << "\n"; }
    std::cout << "\n";

    //======================================================
    // CALCULATE DENSITY MAP
    //======================================================
    // initialise the density maps
    for (int i = 0; i < SIZE_MATRIX; i++) {
        for (int j = 0; j < SIZE_MATRIX; j++) {
            cuda_host_mat[i*SIZE_MATRIX + j] = (int)0;
            openMP_mat[i*SIZE_MATRIX + j] = (int)0;
            seq_mat[i*SIZE_MATRIX + j] = (int)0;
        }
    }
    // find the bounds of the clifford attractor points
    x_range = getMinMax(points_x, NUM_POINTS);
    y_range = getMinMax(points_y, NUM_POINTS);

    // calculate the density map on the GPU in parallel
    if (run_GPU) { 
        std::cout << "Calculating density map in parllel via GPU...\n";
        auto cuda_start = std::chrono::high_resolution_clock::now();
        // Wrapper::
        device_wrapper(x_range.min, x_range.max, y_range.min, y_range.max, 
            NUM_POINTS, points_x, points_y, bytes_points, SIZE_MATRIX, cuda_host_mat, bytes_mat);
        auto cuda_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cuda_elapsed = cuda_end - cuda_start;
        std::cout << "GPU computation elapsed: " << cuda_elapsed.count() << " seconds\n";
        std::cout << "\n";
    }

    // calculate the density map on the CPU in parallel
    if (run_CPU) { 
        std::cout << "Calculating density map in parllel via CPU...\n";
        auto openMp_start = std::chrono::high_resolution_clock::now();
        calcDensity_CPU(x_range.min, x_range.max, y_range.min, y_range.max, 
                        NUM_POINTS, points_x, points_y, SIZE_MATRIX, openMP_mat);
        auto openMP_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> openMP_elapsed = openMP_end - openMp_start;
        std::cout << "CPU computation elapsed: " << openMP_elapsed.count() << " seconds\n";
        std::cout << "\n";
    }

    // calculate the density map on the CPU sequentially
    if (run_SEQ) { 
        std::cout << "Calculating density map sequentially...\n";
        auto seq_start = std::chrono::high_resolution_clock::now();
        calcDensity_SEQ(x_range.min, x_range.max, y_range.min, y_range.max, 
                            NUM_POINTS, points_x, points_y, SIZE_MATRIX, seq_mat);
        auto seq_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> seq_elapsed = seq_end - seq_start;
        std::cout << "Sequential computation elapsed: " << seq_elapsed.count() << " seconds\n";
        std::cout << "\n";
    }

    //======================================================
    // SAVE CLIFFORD ATTRACTOR POINTS
    //======================================================
    if (save_points) {
        std::cout << "Saving Clifford Attractor points...\n";
        std::ofstream mypoints("data/data_points_" + param_set_string + ".txt");
        if (mypoints.is_open()) {
            for (int i = 0; i < NUM_POINTS; i++) { mypoints << points_x[i] << ", " << points_y[i] << "\n"; }
            mypoints.close();
        } else {
            std::cout << "ERROR: Unable to open density file.\n";
            throw std::runtime_error("ERROR: Unable to open density file.\n");
        }
    }
    std::cout << "\n";

    //======================================================
    // SAVE DENSITY MAP DATA
    //======================================================
    std::cout << "Saving density map...\n";

    // save GPU density data
    if (run_GPU && save_density) {
        std::cout << "\t> Saving GPU data...\n";
        std::ofstream mydensity_gpu("data/data_density_gpu_" + param_set_string + ".txt");
        if (mydensity_gpu.is_open()) {
            for (int i = 0; i < SIZE_MATRIX; i++) {
                for (int j = 0; j < SIZE_MATRIX; j++) {
                    if (j < SIZE_MATRIX-1) {
                        mydensity_gpu << cuda_host_mat[i*SIZE_MATRIX + j] << ", ";
                    } else {
                        mydensity_gpu << cuda_host_mat[i*SIZE_MATRIX + j] << "\n";
                    }
                }
            }
            mydensity_gpu.close();
        } else {
            std::cout << "ERROR: Unable to open density file.\n";
            throw std::runtime_error("ERROR: Unable to open density file.\n");
        }
    }

    // save CPU density data
    if (run_CPU && save_density) {
        std::cout << "\t> Saving CPU data...\n";
        std::ofstream mydensity_cpu("data/data_density_cpu_" + param_set_string + ".txt");
        if (mydensity_cpu.is_open()) {
            for (int i = 0; i < SIZE_MATRIX; i++) {
                for (int j = 0; j < SIZE_MATRIX; j++) {
                    if (j < SIZE_MATRIX-1) {
                        mydensity_cpu << openMP_mat[i*SIZE_MATRIX + j] << ", ";
                    } else {
                        mydensity_cpu << openMP_mat[i*SIZE_MATRIX + j] << "\n";
                    }
                }
            }
            mydensity_cpu.close();
        } else {
            std::cout << "ERROR: Unable to open density file.\n";
            throw std::runtime_error("ERROR: Unable to open density file.\n");
        }
    }

    // save sequential density data
    if (run_SEQ && save_density) {
        std::cout << "\t> Saving sequential data...\n";
        std::ofstream mydensity_seq("data/data_density_seq_" + param_set_string + ".txt");
        if (mydensity_seq.is_open()) {
            for (int i = 0; i < SIZE_MATRIX; i++) {
                for (int j = 0; j < SIZE_MATRIX; j++) {
                    if (j < SIZE_MATRIX-1) {
                        mydensity_seq << seq_mat[i*SIZE_MATRIX + j] << ", ";
                    } else {
                        mydensity_seq << seq_mat[i*SIZE_MATRIX + j] << "\n";
                    }
                }
            }
            mydensity_seq.close();
        } else {
            std::cout << "ERROR: Unable to open density file.\n";
            throw std::runtime_error("ERROR: Unable to open density file.\n");
        }
    }

    std::cout << "\n";

    //======================================================
    // RELEASE HOST (CPU) DATA
    //======================================================
    free(points_x);
    free(points_y);
    free(cuda_host_mat);
    free(openMP_mat);
    free(seq_mat);

    // end of program
    std::cout << "Finished program.\n";
    return 0;
}
