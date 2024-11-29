<!--
 g07_Cyclone_Compiling.md

 CaSToRC, The Cyprus Institute

 (c) 2024 The Cyprus Institute

 Contributing Authors:
 Christodoulos Stylianou (c.stylianou@cyi.ac.cy)
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     https://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# **7. Compiling and Running C/C++ Code on Cyclone**

Cyclone provides a robust environment for compiling and running C/C++ applications using various programming models, such as standard single-threaded C/C++ code, OpenMP for multithreading, CUDA for GPU acceleration, MPI for distributed memory parallelism, and hybrid models like MPI+OpenMP or MPI+CUDA.

This guide walks through the process of compiling and running "Hello World" programs for each of these paradigms using Cyclone's available tools and compilers. 

---

## **7.1 Compiling and Running Standard C/C++ Code**

### **7.1.1 Sample Hello World Code (C)**
Save the following as `hello.c`:
```c
#include <stdio.h>
int main() {
    printf("Hello, World from Cyclone!\n");
    return 0;
}
```

### **7.1.2 Compiling with GNU Compiler (GCC)**
```bash
module load GCC/11.3.0
gcc hello.c -o hello
```

### **7.1.3 SLURM Job Script**
Save the following as `run_hello_c.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=hello_c
#SBATCH --output=hello_c.out
#SBATCH --error=hello_c.err
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --account=<your_account>

module load GCC/11.3.0
srun ./hello
```

Submit the job:
```bash
sbatch run_hello_c.slurm
```

### **7.1.4 Compiling and Running with Intel Compiler**
Recompile the code to use Intel Compilers:
```bash
module load intel/2022b
icc hello.c -o hello
```

To run using the Intel Compilers and runtime, update the `run_hello_c.slurm` to use the `intel/2022b` module instead of `GCC/11.3.0`.

---

## **7.2 Compiling and Running OpenMP Code**

OpenMP enables multithreading within a single compute node by parallelizing tasks across multiple cores.

### **7.2.1 Sample OpenMP Hello World Code**
Save the following as `hello_openmp.c`:
```c
#include <stdio.h>
#include <omp.h>
int main() {
    #pragma omp parallel
    {
        printf("Hello from thread %d out of %d\n", omp_get_thread_num(), omp_get_num_threads());
    }
    return 0;
}
```

### **7.2.2 Compiling with GCC**
```bash
module load GCC/11.3.0
gcc -fopenmp hello_openmp.c -o hello_openmp
```

### **7.2.3 SLURM Job Script**
Save the following as `run_openmp.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=openmp
#SBATCH --output=openmp.out
#SBATCH --error=openmp.err
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8       # Runs on 8 threads
#SBATCH --time=00:15:00
#SBATCH --account=<your_account>

module load GCC/11.3.0

export OMP_NUM_THREADS=8
srun ./hello_openmp
```

Submit the job:
```bash
sbatch run_openmp.slurm
```

### **7.2.4 Compiling with Intel Compiler**
```bash
module load intel/2022b
icc -qopenmp hello_openmp.c -o hello_openmp
```

**Important Note**: When Intel Compilers are used instead of `-fopenmp` when compiling for OpenMP Code, `-qopenmp` is used insted.

---

## **7.3 Compiling and Running CUDA Code**

NVIDIA GPUs us CUDA for parallel computations.

### **7.3.1 Sample CUDA Hello World Code**
Save the following as `hello_cuda.cu`:
```cpp
#include <stdio.h>

__global__ void helloFromGPU() {
    printf("Hello, World from GPU thread %d\n", threadIdx.x);
}

int main() {
    helloFromGPU<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}
```
**Important Note**: In practise you should avoid printing from inside a CUDA Kernel (e.g., `helloFromGPU`) as this can slowdown drastically the computations.

### **7.3.2 Compiling CUDA Code**
```bash
module load CUDA/12.1.1
nvcc hello_cuda.cu -o hello_cuda
```

### **7.3.3 SLURM Job Script for CUDA**
Save the following as `run_cuda.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=cuda
#SBATCH --output=cuda.out
#SBATCH --error=cuda.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1        # Runs on 1 GPU
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --account=<your_account>

module load CUDA/12.1.1
srun ./hello_cuda
```

Submit the job:
```bash
sbatch run_cuda.slurm
```

---

## **7.4 Compiling and Running MPI Code**

MPI enables distributed memory parallelism across multiple nodes.

### **7.4.1 Sample MPI Hello World Code**
Save the following as `hello_mpi.c`:
```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("Hello from process %d out of %d\n", rank, size);
    MPI_Finalize();
    return 0;
}
```

### **7.4.2 Compiling**
```bash
module load OpenMPI/4.1.6-GCC-13.2.0
mpicc hello_mpi.c -o hello_mpi
```

### **7.4.3 SLURM Job Script for MPI**
Save the following as `run_mpi.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=mpi
#SBATCH --output=mpi.out
#SBATCH --error=mpi.err
#SBATCH --partition=cpu
#SBATCH --ntasks=4      # Runs on 4 Processes
#SBATCH --time=00:20:00
#SBATCH --account=<your_account>

module load OpenMPI/4.1.6-GCC-13.2.0
srun ./hello_mpi
```

Submit the job:
```bash
sbatch run_mpi.slurm
```

---

## **7.5 Compiling and Running MPI+OpenMP Code**

Hybrid MPI+OpenMP combines distributed and shared memory parallelism.

### **7.5.1 Sample MPI+OpenMP Code**
Save the following as `hello_mpi_openmp.c`:
```c
#include <mpi.h>
#include <omp.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    #pragma omp parallel
    {
        printf("Hello from MPI process %d, thread %d out of %d\n", 
                rank, omp_get_thread_num(), omp_get_num_threads());
    }

    MPI_Finalize();
    return 0;
}
```

### **7.5.2 Compiling**
```bash
module load OpenMPI/4.1.6-GCC-13.2.0
mpicc -fopenmp hello_mpi_openmp.c -o hello_mpi_openmp
```

### **7.5.3 SLURM Job Script for MPI+OpenMP**
Save the following as `run_mpi_openmp.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=mpi_openmp
#SBATCH --output=mpi_openmp.out
#SBATCH --error=mpi_openmp.err
#SBATCH --partition=cpu
#SBATCH --ntasks=4          # 4 Processes
#SBATCH --cpus-per-task=8   # Each with 8 threads/cores
#SBATCH --time=00:30:00
#SBATCH --account=<your_account>

module load OpenMPI/4.1.6-GCC-13.2.0

export OMP_NUM_THREADS=8
srun ./hello_mpi_openmp
```

Submit the job:
```bash
sbatch run_mpi_openmp.slurm
```

---

## **7.6 Compiling and Running MPI+CUDA Code**

This hybrid model uses MPI for inter-node communication and CUDA for GPU acceleration.

### **7.6.1 Sample MPI+CUDA Code**
Save the following as `hello_mpi_cuda.cu`:
```cpp
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void helloFromGPU(int rank) {
    printf("Hello from GPU thread %d of MPI process %d\n", threadIdx.x, rank);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    helloFromGPU<<<1, 10>>>(rank);
    cudaDeviceSynchronize();

    MPI_Finalize();
    return 0;
}
```

### **7.6.2 Compiling**
```bash
module load OpenMPI/4.1.6-GCC-13.2.0 CUDA/12.1.1
mpicc -ccbin=nvcc hello_mpi_cuda.cu -o hello_mpi_cuda
```

### **7.6.3 SLURM Job Script for MPI+CUDA**
Save the following as `run_mpi_cuda.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=mpi_cuda
#SBATCH --output=mpi_cuda.out
#SBATCH --error=mpi_cuda.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1        # 1 GPU per process
#SBATCH --ntasks=4          # 4 Processes
#SBATCH --time=00:30:00
#SBATCH --account=<your_account>

module load OpenMPI/4.1.6-GCC-13.2.0 CUDA/12.1.1 CUDA/12.1.1

srun ./hello_mpi_cuda
```

Submit the job:
```bash
sbatch run_mpi_cuda.slurm
```

## **7.7 Compiling and Running OpenMP+CUDA Code**
This example highlights how OpenMP threads on the CPU can manage multiple CUDA devices, enabling hybrid parallelism. The parallelism allows both CPUs and GPUs to work together on the same node. Compared to the equivalent MPI+CUDA, this approach is usually a good place to start for people that are looking to either increase the CPU or GPU parallelism in their code with minimum code modifications. In general, OpenMP is less intrusive compared to MPI when it comes to code modifications, with the limitation that it can only scale up to a single node.

### **7.7.1 Sample OpenMP+CUDA Code**
Save the following as `hello_openmp_cuda.cu`:
```cpp
#include <omp.h>
#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel to print from GPU threads
__global__ void cuda_hello(int thread_id, int omp_thread) {
    printf("Hello from CUDA thread %d in OpenMP thread %d\n", threadIdx.x, omp_thread);
}

int main() {
    // Total OpenMP threads
    const int total_threads = 40;
    const int threads_per_gpu = 10;

    // OpenMP parallel region
    #pragma omp parallel num_threads(total_threads)
    {
        int omp_thread_id = omp_get_thread_num();

        // Allow only one thread on the socket to launch the kernel
        if(omp_thread_id%10==0){
            // Determine GPU based on thread ID: Map 10 threads to each GPU
            int gpu_id = omp_thread_id / threads_per_gpu; 
            cudaSetDevice(gpu_id); // Set the CUDA device
            
            printf("Hello from OpenMP thread %d managing GPU %d\n", omp_thread_id, gpu_id);

            // Launch a CUDA kernel
            cuda_hello<<<1, 5>>>(omp_thread_id, gpu_id);
            cudaDeviceSynchronize(); // Ensure GPU kernel completion
        }
    }

    return 0;
}
```

### **7.7.2** Compiling
```bash
module load GCC/11.3.0 CUDA/12.1.1
nvcc -Xcompiler -fopenmp hello_openmp_cuda.cu -o hello_openmp_cuda
```
### **7.7.3 SLURM Job Script for OpenMP+CUDA**
Save the following as `run_openmp_cuda.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=openmp_cuda
#SBATCH --output=openmp_cuda.out
#SBATCH --error=openmp_cuda.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4            # Request 4 GPUs
#SBATCH --cpus-per-task=40      # Request 40 CPUs for OpenMP threads
#SBATCH --time=00:10:00
#SBATCH --account=<your_account>

module load GCC/11.3.0 CUDA/12.1.1

# Run the executable
srun ./hello_openmp_cuda
```

Submit the job:
```bash
sbatch run_openmp_cuda.slurm
```

---

## **7.8 Best Practices**
1. **Choose the Right Compiler and Modules**: Use GCC or Intel for C/C++. For GPU-based codes, CUDA must be loaded.
2. **Use Appropriate Partitions**: Use the cpu partition for CPU-only jobs and gpu partition for GPU-enabled jobs.
3. **Compile on Compute Nodes**: The compilation of large libraries/codebases might require a lot of resources, especially when done in parallel (e.g., using `make -j`). To avoid saturating the login node, this process can be done on the compute nodes, so the compilation can be placed in a SLURM file and submited to the scheduler.

---
