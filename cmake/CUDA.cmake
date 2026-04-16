option(CXM_USE_CUDA "Enable CUDA backend" ON)

if (CXM_USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit)
endif ()

if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found: enabling CUDA backend")
    set(CXM_CUDA_AVAILABLE TRUE)
else()
    message(STATUS "CUDA not found: disabling CUDA backend")
    set(CXM_CUDA_AVAILABLE FALSE)
endif()

if(CXM_CUDA_AVAILABLE)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()