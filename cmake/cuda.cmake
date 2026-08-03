option(CXM_USE_CUDA "Enable CUDA backend" ON)
set(CXM_IS_CUDA_AVAILABLE FALSE)

if(CXM_USE_CUDA)
    find_package(CUDAToolkit QUIET)

    if(CUDAToolkit_FOUND)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON) # Ensures that the specified standard is mandatory.
        set(CMAKE_CUDA_EXTENSIONS OFF)       # Disables compiler-specific extensions and enforces the pure C++20 standard.

        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES native)
        endif()

        enable_language(CUDA)
        set(CXM_IS_CUDA_AVAILABLE TRUE)

        message(STATUS "CXM: CUDA backend enabled successfully (Arch: ${CMAKE_CUDA_ARCHITECTURES})")
    else()
        message(STATUS "CXM: CUDA backend disabled (CUDA Toolkit not found)")
    endif()
else()
    message(STATUS "CXM: CUDA backend disabled by user")
endif()