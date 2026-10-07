#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        cudaError_t err = (call);                                                               \
        if (err != cudaSuccess) {                                                               \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                                                 \
            std::exit(EXIT_FAILURE);                                                            \
        }                                                                                       \
    } while (0)

__global__ void naiveGemmKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // A(row, k) -> row * K + k
            // B(k, col) -> k * N + col
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

bool nearlyEqual(float actual, float expected, float atol = 1e-4f, float rtol = 1e-3f) {
    return std::abs(actual - expected) <= atol + rtol * std::abs(expected);
}

bool validateResult(const std::vector<float>& gpu, const std::vector<float>& cpu) {
    if (gpu.size() != cpu.size()) {
        std::cerr << "Size mismatch: GPU=" << gpu.size() << ", CPU = " << cpu.size() << '\n';

        return false;
    }

    for (size_t i = 0; i < gpu.size(); i++) {
        if (!nearlyEqual(gpu[i], cpu[i])) {
            std::cerr << "Mismatch at index: " << i << ": GPU = " << gpu[i] << ": CPU = " << cpu[i]
                      << std::endl;
            return false;
        }
    }
    return true;
}

void cpuGemmReference(const std::vector<float>& h_A, const std::vector<float>& h_B,
                      std::vector<float>& h_C_ref, const int M, const int N, const int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            h_C_ref[i * N + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    bool validationMode = false;
    if (argc == 2 && std::string(argv[1]) == "--validate") {
        validationMode = true;
    }

    if (!validationMode && argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <blockX> <blockY> or --validate\n";
        return 1;
    }

    int M, K, N;
    int blockX, blockY;
    int warmupRuns, benchRuns;

    if (validationMode) {
        warmupRuns = 1;
        benchRuns = 1;

        M = 127;
        K = 193;
        N = 251;

        blockX = 16;
        blockY = 16;
    } else {
        warmupRuns = 5;
        benchRuns = 100;

        M = 2048;
        K = 2048;
        N = 2048;

        blockX = std::atoi(argv[1]);
        blockY = std::atoi(argv[2]);
    }

    std::cout << "=== Naive GEMM Benchmark (M=" << M << ", N=" << N << ", K=" << K
              << ") ===" << std::endl;
    std::cout << "Mode: " << (validationMode ? "Validation" : "Benchmark") << '\n';
    std::cout << "<blockX> <blockY> :" << blockX << " " << blockY << std::endl;

    if (blockX <= 0 || blockY <= 0) {
        std::cerr << "Block dimensions must be positive\n";
        return 1;
    }

    if (blockX * blockY > 1024) {
        std::cerr << "Too many threads per block\n";
        return 1;
    }

    size_t sizeA = static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(float);
    size_t sizeB = static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(float);
    size_t sizeC = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float);

    // Allocate host memory
    std::vector<float> h_A(static_cast<size_t>(M) * K);      // matrix A
    std::vector<float> h_B(static_cast<size_t>(K) * N);      // matrix B
    std::vector<float> h_C(static_cast<size_t>(M) * N);      // matrix C
    std::vector<float> h_C_ref(static_cast<size_t>(M) * N);  // matrix C for CPU

    for (size_t i = 0; i < h_A.size(); i++) {
        h_A[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }

    for (size_t i = 0; i < h_B.size(); i++) {
        h_B[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }

    // Allocate device memory
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));

    // Copy host memory to device memory
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice));

    // Number of threads per block
    dim3 threads(blockX, blockY);

    // Number of block per grid
    int blockPerGridX = (N + blockX - 1) / blockX;
    int blockPerGridY = (M + blockY - 1) / blockY;
    dim3 blocks(blockPerGridX, blockPerGridY);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up runs
    for (int i = 0; i < warmupRuns; i++) {
        naiveGemmKernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure kernel execution time
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < benchRuns; i++) {
        naiveGemmKernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float naiveTotalTimeMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(&naiveTotalTimeMs, start, stop));
    float naiveAvgTimeMs = naiveTotalTimeMs / benchRuns;

    // Copy device memory to host
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost));

    if (validationMode) {
        cpuGemmReference(h_A, h_B, h_C_ref, M, N, K);

        bool correct = validateResult(h_C, h_C_ref);
        std::cout << "Validation: " << (correct ? "PASSED" : "FAILED") << '\n';
    } else {
        // GEMM flops: 2 * M * N * K
        double totalFlops =
            2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        double naiveGFlops = totalFlops / (naiveAvgTimeMs * 1e6);

        std::cout << "\n---------------- Performance Results ----------------" << std::endl;

        std::cout << "Naive GEMM Time : " << naiveAvgTimeMs << " ms | Performance: " << naiveGFlops
                  << " GFLOPS" << std::endl;

        std::cout << "-----------------------------------------------------" << std::endl;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
