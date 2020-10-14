all: 
    nvcc -g .\src\main.cpp .\src\calc_density_CPU.cpp .\src\calc_density_SEQ.cpp .\src\calc_density_GPU.cu  -I.\include\ -Xcompiler -openmp -o program
