cmake_minimum_required(VERSION 3.20)
project(patchcore_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(patchcore_native SHARED
    src/patchcore_native.cpp
)

target_include_directories(patchcore_native PUBLIC include)
target_compile_definitions(patchcore_native PRIVATE PCN_EXPORTS)

if(MSVC)
    target_compile_options(patchcore_native PRIVATE /O2 /fp:fast /arch:AVX2)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(patchcore_native PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endif()
else()
    target_compile_options(patchcore_native PRIVATE -O3 -ffast-math -mavx2 -mfma)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(patchcore_native PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

set_target_properties(patchcore_native PROPERTIES
    OUTPUT_NAME "patchcore_native"
)

foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${CONFIG} CONFIG_UPPER)
    set_target_properties(patchcore_native PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_BINARY_DIR}/bin"
        LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_BINARY_DIR}/bin"
    )
endforeach()

if(NOT CMAKE_CONFIGURATION_TYPES)
    set_target_properties(patchcore_native PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()
