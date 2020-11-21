#include "calc_density_GPU.cuh"

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

// compute clifford attractor points via GPU
__global__
void calcDensity_GPU(double x_min, double x_max, double y_min, double y_max, 
                    double* x, double* y, int N, int M, int* A) {
    int point_id = blockDim.x*blockIdx.x + threadIdx.x;
    int tmp_col = -1;
    int tmp_row = -1;
    double cell_width, cell_height;
    if (point_id < N) {
        // save dimensions of cell
        cell_width = (x_max - x_min)/(double)M;
        cell_height = (y_max - y_min)/(double)M;
        // find the cell in which the point falls
        tmp_col = MIN((x[point_id] - x_min)/cell_width, M-1);
        tmp_row = MIN((y[point_id] - y_min)/cell_height, M-1);
        // atomically increment cell count
        if (tmp_col >= 0 && tmp_row >= 0) { atomicAdd(&A[tmp_row*M + tmp_col], 1); }
    }
}

// namespace Wrapper {
    void device_wrapper(double x_min, double x_max, double y_min, double y_max, 
                        int NUM_POINTS, double* host_x, double* host_y, unsigned int bytes_points, 
                        int SIZE_MATRIX, int* host_mat, unsigned int bytes_mat) {
        //======================================================
        // ALOCATE & INITIALISE DEVICE DATA
        //======================================================
        // booleans to check if cuda played nicely
        cudaError_t result_x, result_y, result_mat, sync_check;
        // allocate space on device for vectors
        double *device_x;
        double *device_y;
        result_x = cudaMalloc((void**)&device_x, bytes_points);
        result_y = cudaMalloc((void**)&device_y, bytes_points);
        if ((result_x != 0) && (result_y != 0)) {
            std::cout << "\t> " << "ERROR: Failed to allocate device memory.\n";
            throw std::runtime_error("ERROR: Failed to allocate device memory.\n");
        } else { std::cout << "\t> " "Successfully allocated: " << bytes_points << " bytes of data on the device.\n"; }
        // copy host density matrix to device
        result_x = cudaMemcpy(device_x, host_x, bytes_points, cudaMemcpyHostToDevice);
        result_y = cudaMemcpy(device_y, host_y, bytes_points, cudaMemcpyHostToDevice);
        if ((result_x != 0) && (result_y != 0)) {
            std::cout << "\t> " "ERROR: Failed to copy data to device memory.\n";
            throw std::runtime_error("ERROR: Failed to copy data to device memory.\n");
        } else { std::cout << "\t> " "Successfully coppied data to device memory.\n"; }
        // allocate space on device for density matrix
        int *device_mat;
        result_mat = cudaMalloc((void**)&device_mat, bytes_mat);
        if (result_mat != 0) {
            std::cout << "\t> " "ERROR: Failed to allocate device memory.\n";
            throw std::runtime_error("ERROR: Failed to allocate device memory.\n");
        } else { std::cout << "\t> " "Successfully allocated: " << bytes_mat << " bytes of data on the device.\n"; }
        // copy host density matrix to device
        result_mat = cudaMemcpy(device_mat, host_mat, bytes_mat, cudaMemcpyHostToDevice);
        if (result_mat != 0) {
            std::cout << "\t> " "ERROR: Failed to copy data to device memory.\n";
            throw std::runtime_error("ERROR: Failed to copy data to device memory.\n");
        } else { std::cout << "\t> " "Successfully coppied data to device memory.\n"; }

        //======================================================
        // RUN KERNEL ON DEVICE
        //======================================================
        // Number of threads in each thread block
        int threadsPerBlock = 32 * 4 * 4;
        // Number of thread blocks in grid
        int numBlocks = (NUM_POINTS + threadsPerBlock - 1) / threadsPerBlock;
        std::cout << "\t> " "Number of blocks: " << numBlocks << "\n";
        std::cout << "\t> " "Number of treads/block: " << threadsPerBlock << "\n";
        // Execute the kernel
        std::cout << "\t> " "Calculating density of points...\n";
        calcDensity_GPU<<<numBlocks, threadsPerBlock>>>(x_min, x_max, y_min, y_max, device_x, device_y, NUM_POINTS, SIZE_MATRIX, device_mat);
        sync_check = cudaDeviceSynchronize();
        if (sync_check != 0) {
            std::cout << "\t> " "ERROR: Kernel failed with code " << sync_check << ".\n";
            throw std::runtime_error("ERROR: Kernel failed.\n");
        }

        //======================================================
        // COPY DEVICE DATA BACK TO HOST
        //======================================================
        // Copy density matrix back to host
        result_mat = cudaMemcpy(host_mat, device_mat, bytes_mat, cudaMemcpyDeviceToHost);
        if (result_mat != 0) {
            std::cout << "\t> " "ERROR: Failed to copy data to host memory.\n";
            throw std::runtime_error("ERROR: Failed to copy data to host memory.\n");
        } else { std::cout << "\t> " "Successfully coppied data to host memory.\n"; }

        // Release device memory
        cudaFree(device_x);
        cudaFree(device_y);
        cudaFree(device_mat);
    }
// }
