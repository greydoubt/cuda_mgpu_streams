// Copy/Compute Overlap with Multiple Streams for Multiple GPUs

// For each GPU, we will perform copy/compute overlap in multiple non-default streams. This technique is very similar as that with only one GPU, only we must do it while looping over each GPU, and, take some additional care with indexing into the data. Work through this section slowly:
#include <iostream>
#include <cstdint>
#include <algorithm>

template<typename T>
void process_data(T* data_cpu, T** data_gpu, uint64_t num_entries, uint64_t num_gpus, uint64_t num_streams, uint64_t gpu_chunk_size, uint64_t stream_chunk_size, dim3 grid, dim3 block, cudaStream_t** streams) {
    for (uint64_t gpu = 0; gpu < num_gpus; gpu++) {
        // ...set device as active.
        cudaSetDevice(gpu);
        // For each stream (on each GPU)...
        for (uint64_t stream = 0; stream < num_streams; stream++) {

            // Calculate index offset for this stream's chunk of data within the GPU's chunk of data...
            const uint64_t stream_offset = stream_chunk_size*stream;
            
            // ...get the lower index within all data, and width of this stream's data chunk...
            const uint64_t lower = gpu_chunk_size*gpu+stream_offset;
            const uint64_t upper = std::min(lower+stream_chunk_size, num_entries);
            const uint64_t width = upper-lower;

            // ...perform async HtoD memory copy...
            cudaMemcpyAsync(data_gpu[gpu]+stream_offset, // This stream's data within this GPU's data.
                            data_cpu+lower,              // This stream's data within all CPU data.
                            sizeof(T)*width,             // This stream's chunk size worth of data.
                            cudaMemcpyHostToDevice,
                            streams[gpu][stream]);       // Using this stream for this GPU.

            kernel<<<grid, block, 0, streams[gpu][stream]>>>    // Using this stream for this GPU.
                (data_gpu[gpu]+stream_offset,                   // This stream's data within this GPU's data.
                width);                                         // This stream's chunk size worth of data.

            cudaMemcpyAsync(data_cpu+lower,              // This stream's data within all CPU data.
                            data_gpu[gpu]+stream_offset, // This stream's data within this GPU's data.
                            sizeof(T)*width,
                            cudaMemcpyDeviceToHost,
                            streams[gpu][stream]);       // Using this stream for this GPU.
        }
    }
}

int main() {
    // Example usage
    constexpr uint64_t num_entries = 1000;
    constexpr uint64_t num_gpus = 1;
    constexpr uint64_t num_streams = 1;
    constexpr uint64_t gpu_chunk_size = 100;
    constexpr uint64_t stream_chunk_size = 100;
    dim3 grid(1);
    dim3 block(1);

    // Allocate and initialize data_cpu and data_gpu
    uint64_t* data_cpu = new uint64_t[num_entries];
    uint64_t** data_gpu = new uint64_t*[num_gpus];
    cudaStream_t** streams = new cudaStream_t*[num_gpus];
    for (uint64_t i = 0; i < num_gpus; i++) {
        cudaMalloc(&data_gpu[i], sizeof(uint64_t) * num_entries);
        cudaStream_t* temp_streams = new cudaStream_t[num_streams];
        for (uint64_t j = 0; j < num_streams; j++) {
            cudaStreamCreate(&temp_streams[j]);
        }
        streams[i] = temp_streams;
    }

    // Call the templated function
    process_data<uint64_t>(data_cpu, data_gpu, num_entries, num_gpus, num_streams, gpu_chunk_size, stream_chunk_size, grid, block, streams);

    // Free allocated memory
    delete[] data_cpu;
    for (uint64_t i = 0; i < num_gpus; i++) {
        cudaFree(data_gpu[i]);
        for (uint64_t j = 0; j < num_streams; j++) {
            cudaStreamDestroy(streams[i][j]);
        }
        delete[] streams[i];
    }
    delete[] data_gpu;
    delete[] streams;

    return 0;
}
