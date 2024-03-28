// Data Chunk Sizes for Multiple Streams on Multiple GPUs

// Indexing into global data becomes even more tricky when using multiple non-default streams with multiple GPUs. It can be helpful to define data chunk sizes for a single stream, as well as data chunk sizes for an entire GPU:

// Each stream needs num_entries/num_gpus/num_streams data. We use round up division for
// reasons previously discussed.
const uint64_t stream_chunk_size = sdiv(sdiv(num_entries, num_gpus), num_streams);

// It will be helpful to also to have handy the chunk size for an entire GPU.
const uint64_t gpu_chunk_size = stream_chunk_size*num_streams;

