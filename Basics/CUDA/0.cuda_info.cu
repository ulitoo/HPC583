#include <iostream>

// Helper function to convert compute capability to CUDA cores
int _ConvertSMVer2Cores(int major, int minor)
{
    // Refer to the CUDA Toolkit documentation for compute capability details:
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
    int cores;
    switch ((major << 4) + minor) {
        case 0x20: // Compute Capability 2.0
            cores = 32;
            break;
        case 0x30: // Compute Capability 3.0
            cores = 192;
            break;
        case 0x32: // Compute Capability 3.2
            cores = 192;
            break;
        case 0x35: // Compute Capability 3.5
            cores = 192;
            break;
        case 0x37: // Compute Capability 3.7
            cores = 192;
            break;
        case 0x50: // Compute Capability 5.0
            cores = 128;
            break;
        case 0x52: // Compute Capability 5.2
            cores = 128;
            break;
        case 0x53: // Compute Capability 5.3
            cores = 128;
            break;
        case 0x60: // Compute Capability 6.0
            cores = 64;
            break;
        case 0x61: // Compute Capability 6.1
            cores = 128;
            break;
        case 0x62: // Compute Capability 6.2
            cores = 128;
            break;
        case 0x70: // Compute Capability 7.0
            cores = 64;
            break;
        case 0x72: // Compute Capability 7.2
            cores = 64;
            break;
        case 0x75: // Compute Capability 7.5
            cores = 64;
            break;
        default:
            cores = -1; // Unknown compute capability
            break;
    }
    return cores;
}

int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "No CUDA-enabled devices found" << std::endl;
        return -1;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
        std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "CUDA Cores: " << deviceProp.multiProcessorCount * _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) << std::endl;
        std::cout << "Clock Rate: " << deviceProp.clockRate / 1000 << " MHz" << std::endl << std::endl;
    }

    return 0;
}
