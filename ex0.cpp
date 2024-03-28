// Creating Multiple Streams for Multiple GPUs

// When using multiple non-default streams on multiple GPUs, we can store them in a 2D array, with each row containing the streams for a single GPU:

cudaStream_t streams[num_gpus][num_streams]; // 2D array containing number of streams for each GPU.

// For each available GPU...
for (uint64_t gpu = 0; gpu < num_gpus; gpu++) {
    // ...set as active device...
    cudaSetDevice(gpu);
    for (uint64_t stream = 0; stream < num_streams; stream++)
        // ...create and store its number of streams.
        cudaStreamCreate(&streams[gpu][stream]);
}

