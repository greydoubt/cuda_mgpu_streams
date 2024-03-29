// Copy/Compute Overlap with Multiple Streams for Multiple GPUs

// For each GPU, we will perform copy/compute overlap in multiple non-default streams. This technique is very similar as that with only one GPU, only we must do it while looping over each GPU, and, take some additional care with indexing into the data. Work through this section slowly:

// For each GPU...
for (uint64_t gpu = 0; gpu < num_gpus; gpu++) {
    // ...set device as active.
    cudaSetDevice(gpu);
    // For each stream (on each GPU)...
    for (uint64_t stream = 0; stream < num_streams; stream++) {

        // Calculate index offset for this stream's chunk of data within the GPU's chunk of data...
        const uint64_t stream_offset = stream_chunk_size*stream;
        
        // ...get the lower index within all data, and width of this stream's data chunk...
        const uint64_t lower = gpu_chunk_size*gpu+stream_offset;
        const uint64_t upper = min(lower+stream_chunk_size, num_entries);
        const uint64_t width = upper-lower;

        // ...perform async HtoD memory copy...
        cudaMemcpyAsync(data_gpu[gpu]+stream_offset, // This stream's data within this GPU's data.
                        data_cpu+lower,              // This stream's data within all CPU data.
                        sizeof(uint64_t)*width,      // This stream's chunk size worth of data.
                        cudaMemcpyHostToDevice,
                        streams[gpu][stream]);       // Using this stream for this GPU.

        kernel<<<grid, block, 0, streams[gpu][stream]>>>    // Using this stream for this GPU.
            (data_gpu[gpu]+stream_offset,                   // This stream's data within this GPU's data.
             width);                                        // This stream's chunk size worth of data.

        cudaMemcpyAsync(data_cpu+lower,              // This stream's data within all CPU data.
                        data_gpu[gpu]+stream_offset, // This stream's data within this GPU's data.
                        sizeof(uint64_t)*width,
                        cudaMemcpyDeviceToHost,
                        streams[gpu][stream]);       // Using this stream for this GPU.
    }
}

