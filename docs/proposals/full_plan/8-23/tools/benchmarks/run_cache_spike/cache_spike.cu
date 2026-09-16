#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t status = (call); if (status != cudaSuccess) { std::cerr << cudaGetErrorString(status) << " at " << __LINE__ << "\n"; std::exit(1); } } while (0)

__global__ void busy_kernel(float *values, std::size_t count, int iterations) {
    std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float value = values[index];
    for (int iteration = 0; iteration < iterations; ++iteration) {
        value = fmaf(value, 1.0000001f, 0.0000001f);
    }
    values[index] = value;
}

struct Timing {
    std::uint64_t h2d_ns;
    std::uint64_t compute_ns;
    std::uint64_t overlap_ns;
    std::uint64_t unhidden_ns;
};

Timing measure(std::size_t payload_bytes, bool copy, bool overlap, std::uint8_t *host, std::uint8_t *device, float *compute_values, std::size_t compute_count, cudaStream_t copy_stream, cudaStream_t compute_stream) {
    cudaEvent_t start, copy_done, compute_start, compute_done;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&copy_done));
    CUDA_CHECK(cudaEventCreate(&compute_start));
    CUDA_CHECK(cudaEventCreate(&compute_done));
    CUDA_CHECK(cudaEventRecord(start, nullptr));
    CUDA_CHECK(cudaStreamWaitEvent(copy_stream, start));
    CUDA_CHECK(cudaStreamWaitEvent(compute_stream, start));
    if (copy) CUDA_CHECK(cudaMemcpyAsync(device, host, payload_bytes, cudaMemcpyHostToDevice, copy_stream));
    CUDA_CHECK(cudaEventRecord(copy_done, copy_stream));
    if (!overlap && copy) CUDA_CHECK(cudaStreamWaitEvent(compute_stream, copy_done));
    CUDA_CHECK(cudaEventRecord(compute_start, compute_stream));
    busy_kernel<<<256, 256, 0, compute_stream>>>(compute_values, compute_count, 16384);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(compute_done, compute_stream));
    CUDA_CHECK(cudaEventSynchronize(copy_done));
    CUDA_CHECK(cudaEventSynchronize(compute_done));
    float copy_ms = 0.0f, compute_start_ms = 0.0f, compute_done_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&copy_ms, start, copy_done));
    CUDA_CHECK(cudaEventElapsedTime(&compute_start_ms, start, compute_start));
    CUDA_CHECK(cudaEventElapsedTime(&compute_done_ms, start, compute_done));
    const double overlap_ms = std::max(0.0, std::min<double>(copy_ms, compute_done_ms) - std::max<double>(0.0, compute_start_ms));
    const double compute_ms = std::max(0.0, static_cast<double>(compute_done_ms - compute_start_ms));
    const double unhidden_ms = std::max(0.0, static_cast<double>(copy_ms) - overlap_ms);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(copy_done));
    CUDA_CHECK(cudaEventDestroy(compute_start));
    CUDA_CHECK(cudaEventDestroy(compute_done));
    return {
        copy ? static_cast<std::uint64_t>(std::llround(copy_ms * 1.0e6)) : 0,
        static_cast<std::uint64_t>(std::llround(compute_ms * 1.0e6)),
        copy ? static_cast<std::uint64_t>(std::llround(overlap_ms * 1.0e6)) : 0,
        copy ? static_cast<std::uint64_t>(std::llround(unhidden_ms * 1.0e6)) : 0,
    };
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: cache_spike EXPERT_BYTES\n";
        return 2;
    }
    const std::size_t expert_bytes = std::strtoull(argv[1], nullptr, 10);
    const std::size_t maximum_bytes = 64ULL * 1024ULL * 1024ULL;
    std::uint8_t *host = nullptr, *device = nullptr;
    float *compute_values = nullptr;
    CUDA_CHECK(cudaHostAlloc(&host, maximum_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaMalloc(&device, maximum_bytes));
    const std::size_t compute_count = 256ULL * 256ULL;
    CUDA_CHECK(cudaMalloc(&compute_values, compute_count * sizeof(float)));
    for (std::size_t index = 0; index < maximum_bytes; ++index) host[index] = static_cast<std::uint8_t>((index * 131 + 17) & 0xff);
    CUDA_CHECK(cudaMemset(device, 0, maximum_bytes));
    CUDA_CHECK(cudaMemset(compute_values, 1, compute_count * sizeof(float)));
    cudaStream_t copy_stream, compute_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));

    struct Case { std::string id; std::size_t bytes; bool copy; bool overlap; };
    const std::vector<Case> cases = {
        {"control-8mib-cold", 8ULL * 1024ULL * 1024ULL, true, false},
        {"control-64mib-cold", 64ULL * 1024ULL * 1024ULL, true, false},
        {"manifest-expert-cold", expert_bytes, true, false},
        {"manifest-expert-hit", expert_bytes, false, false},
        {"manifest-expert-evict-load", expert_bytes, true, true},
    };
    std::cout << "case_id,payload_bytes,h2d_duration_ns,compute_duration_ns,overlap_duration_ns,unhidden_transfer_ns,repetition_index\n";
    for (const auto &item : cases) {
        for (int iteration = -5; iteration < 30; ++iteration) {
            Timing timing = measure(item.bytes, item.copy, item.overlap, host, device, compute_values, compute_count, copy_stream, compute_stream);
            if (iteration >= 0) {
                std::cout << item.id << ',' << item.bytes << ',' << timing.h2d_ns << ',' << timing.compute_ns << ',' << timing.overlap_ns << ',' << timing.unhidden_ns << ',' << iteration << '\n';
            }
        }
    }
    CUDA_CHECK(cudaStreamDestroy(copy_stream));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaFree(compute_values));
    CUDA_CHECK(cudaFree(device));
    CUDA_CHECK(cudaFreeHost(host));
    return 0;
}
