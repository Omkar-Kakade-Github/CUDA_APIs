# CUDA API 
> Includes cuBLAS, cuDNN, cuBLASmp

## Error Checking (API Specific)

- cuBLAS for example

```cpp
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
```

- cuDNN example

```cpp
#define CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudnnGetErrorString(status)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
```

- The need for error checking goes as follows: you have a context for a CUDA API call that you configure, then you call the operation, then you check the status of the operation by passing the API call into the "call" field in the macro. If it returns successful your code will continue running as expected. If it fails, you will get a descriptive error message instead of just a segmentation fault or silently incorrect result.
- There are obviously more error checking macros for other CUDA APIs, but these are the most common ones (needed for this course).
- Consider reading this guide here -> [Proper CUDA Error Checking](https://leimao.github.io/blog/Proper-CUDA-Error-Checking/)


## Matrix Multiplication
- cuDNN implicitly supports matmul through specific convolution and deep learning operations but isn't presented as one of the main features of cuDNN
- You'll be best off using the deep learning linear algebra operations in cuBLAS for matrix multiplication since it has wider coverage and is tuned for high throughput matmul
> Side notes (present to show that its not that hard to transfer knowledge of, say, cuDNN to cuFFT with the way you configure and call an operation)

## Resources:
- [CUDA Library Samples](https://github.com/NVIDIA/CUDALibrarySamples)
