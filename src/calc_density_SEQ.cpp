#include "calc_density_SEQ.h"

void calcDensity_SEQ(double x_min, double x_max, double y_min, double y_max, 
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
            // check if the cell has been found
            if ((tmp_row > 0) && (tmp_col > 0)) { break; }
        }
        // atomically increment cell count
        if (tmp_col > 0 && tmp_row > 0) { 
            A[tmp_row*M + tmp_col]++; 
        }
    }
}
