# Using this min version for now
cmake_minimum_required(VERSION 3.1)

project(cb_porting)

# Needed for newer challenges
enable_language(ASM)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

# Default flags for everything
add_compile_options(
    -fno-builtin
    -Wno-int-to-pointer-cast
    -g3
    -m32
)

# Link everything 32-bit (until we have a 64-bit option)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -m32")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -m32")

# Linker options
# Dynamic by default
option(BUILD_SHARED_LIBS "" ON)
if(BUILD_STATIC_LIBS AND UNIX AND NOT APPLE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static -Wl,--allow-multiple-definition")
    SET(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
    SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")
endif()

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
    add_definitions(-DLINUX)
endif()
if(APPLE)
    add_definitions(-DAPPLE)
endif()
if(WIN32)
    add_definitions(-DWIN)
endif()

set(WITEVAL False)
if($ENV{WITEVAL})
    set(WITEVAL True)
    message("using WIT evaluation compiler settings")

    set(CMAKE_ASM_COMPILER clang)
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)

    set(WIT_LINKER ld-wit.py)

    # see: http://stackoverflow.com/questions/1867745/cmake-use-a-custom-linker
    # https://cmake.org/pipermail/cmake/2014-August/058268.html
    set(WIT_C_LINK_EXECUTABLE "ld-wit.py <FLAGS> <CMAKE_C_LINK_FLAGS> <LINK_FLAGS> --XXX -o <TARGET> <LINK_LIBRARIES> --objects <OBJECTS> --objects-end")
    set(WIT_CXX_LINK_EXECUTABLE "ld-wit.py <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> --XXX -o <TARGET> <LINK_LIBRARIES> --cxx --objects <OBJECTS> --objects-end")
    set(WIT_LINK_EXECUTABLE "ld-wit.py <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> --XXX -o <TARGET> <LINK_LIBRARIES> --objects <OBJECTS> --objects-end --link-cmd ")
    set(NOWIT_LINK_EXECUTABLE "ld-wit.py --skipwit <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> --XXX -o <TARGET> <LINK_LIBRARIES> --objects <OBJECTS> --objects-end --link-cmd ")
    #set(CMAKE_C_LINK_EXECUTABLE "${WIT_C_LINK_EXECUTABLE}")
    #set(CMAKE_CXX_LINK_EXECUTABLE "${WIT_CXX_LINK_EXECUTABLE}")
endif()

# both safe-stack and CFI link to pthread
find_package(Threads)

if($ENV{SAFESTACKEVAL})
    set(SAFESTACKEVAL True)
endif()

if($ENV{CFIEVAL})
    set(CFIEVAL True)
endif()

if($ENV{CPIEVAL})
    set(CPIEVAL True)
    message("CPI enabled. Disabling incompatible build types!")
    set(CFIEVAL false)
    # this is upstream safestack, CPI also uses same safestack
    set(SAFESTACKEVAL false)
    set(WITEVAL false)

    set(CPIPATH $ENV{CPIPATH})
    message("CPI build located at ${CPIPATH}")

    set(CPI_CLANG "${CPIPATH}/build/bin/clang")
    set(CPI_CLANGPP "${CPIPATH}/build/bin/clang++")
    
    set(CMAKE_ASM_COMPILER ${CPI_CLANG})
    set(CMAKE_C_COMPILER ${CPI_CLANG})
    set(CMAKE_CXX_COMPILER ${CPI_CLANGPP})

else()

    add_compile_options(-Wno-writable-strings)
endif()

# Build all libraries
include_directories(include)
add_subdirectory(include)

function(build path target)

    # Gather all sources
    aux_source_directory(${path}/src cb_src)
    aux_source_directory(${path}/lib cb_lib)
    aux_source_directory(${path}/include cb_inc)
    set(cb_all_src ${cb_lib} ${cb_src} ${cb_inc})

    add_executable(${target} ${cb_all_src})
    target_include_directories(${target} PUBLIC ${path}/lib ${path}/src ${path}/include)
    target_link_libraries(${target} LINK_PUBLIC cgc)

    add_executable(${target}_patched ${cb_all_src})
    target_include_directories(${target}_patched PUBLIC ${path}/lib ${path}/src ${path}/include)
    # Add patched defines
    target_compile_definitions(${target}_patched PUBLIC "${patch_defs}")
    target_link_libraries(${target}_patched LINK_PUBLIC cgc)

    if(${WITEVAL})
        message("Building ${target} with WIT enabled")

        add_executable(${target}_wit ${cb_all_src})
        target_include_directories(${target}_wit PUBLIC ${path}/lib ${path}/src ${path}/include)
        target_link_libraries(${target}_wit LINK_PUBLIC cgc)
        target_compile_options("${target}_wit" PUBLIC -emit-llvm -c)
        set_target_properties(${target}_wit
                              PROPERTIES
                              RULE_LAUNCH_LINK "${WIT_LINK_EXECUTABLE}" )

        add_executable(${target}_skipwit ${cb_all_src})
        target_include_directories(${target}_skipwit PUBLIC ${path}/lib ${path}/src ${path}/include)
        target_link_libraries(${target}_skipwit LINK_PUBLIC cgc)
        target_compile_options("${target}_skipwit" PUBLIC -emit-llvm -c)
        set_target_properties(${target}_skipwit
                              PROPERTIES
                              RULE_LAUNCH_LINK "${NOWIT_LINK_EXECUTABLE}" )
    endif()

    if(${SAFESTACKEVAL})
        message("Building ${target} with safe-stack enabled")
        set(variant ${target}_safestack)

        add_executable(${variant} ${cb_all_src})
        target_include_directories(${variant} PUBLIC ${path}/lib ${path}/src ${path}/include)
        target_link_libraries(${variant} LINK_PUBLIC cgc)
        execute_process(COMMAND llvm-config --libdir
                        OUTPUT_VARIABLE llvmlibdir
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND llvm-config --version
                        OUTPUT_VARIABLE llvmversion
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
        set(safestack_rt ${llvmlibdir}/clang/${llvmversion}/lib/linux/libclang_rt.safestack-i386.a)
        target_link_libraries(${variant} LINK_PUBLIC
                              ${safestack_rt}
                              ${CMAKE_THREAD_LIBS_INIT}
                              ${CMAKE_DL_LIBS})
        target_compile_options("${variant}" PUBLIC -fsanitize=safe-stack)
    endif()

    if(${CFIEVAL})
        message("Building ${target} with CFI enabled")

        set(variant ${target}_cfi)
        add_executable(${variant} ${cb_all_src})
        target_include_directories(${variant} PUBLIC ${path}/lib ${path}/src ${path}/include)
        target_link_libraries(${variant} LINK_PUBLIC cgc)
        target_compile_options("${variant}" PUBLIC -flto -fvisibility=hidden -fsanitize=cfi -fno-sanitize-trap=all)
        execute_process(COMMAND llvm-config --libdir
                        OUTPUT_VARIABLE llvmlibdir
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND llvm-config --version
                        OUTPUT_VARIABLE llvmversion
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
        set(cfi_rt
            ${llvmlibdir}/clang/${llvmversion}/lib/linux/libclang_rt.ubsan_standalone-i386.a)
        target_link_libraries(${variant} LINK_PUBLIC
                              ${cfi_rt}
                              ${CMAKE_THREAD_LIBS_INIT}
                              ${CMAKE_DL_LIBS}
                              rt)
        set_target_properties(${variant}
            PROPERTIES
            LINK_FLAGS " -flto -fuse-ld=/usr/bin/ld.gold ")
    endif()

    if(${CPIEVAL})
        message("Building ${target} with CPI/CPS enabled")

        set(variant ${target}_cpi)
        add_executable(${variant} ${cb_all_src})
        target_include_directories(${variant} PUBLIC ${path}/lib ${path}/src ${path}/include)
        target_link_libraries(${variant} LINK_PUBLIC cgc m)
        target_compile_options("${variant}" PUBLIC -flto )
        set_target_properties(${variant}
            PROPERTIES
            LINK_FLAGS " -flto -fuse-ld=/usr/bin/ld.gold -fcpi ")

        set(variant ${target}_cps)
        add_executable(${variant} ${cb_all_src})
        target_include_directories(${variant} PUBLIC ${path}/lib ${path}/src ${path}/include)
        target_link_libraries(${variant} LINK_PUBLIC cgc m)
        target_compile_options("${variant}" PUBLIC -flto )
        set_target_properties(${variant}
            PROPERTIES
            LINK_FLAGS " -flto -fuse-ld=/usr/bin/ld.gold -fcps ")
    endif()

endfunction(build)


function(buildPOV path cb pov)
    # Unique target name, i.e. "Palindrome2_pov_1",
    # executable name kept as "pov_#.pov"
    set(target "${cb}_${pov}")

    # Gather sources
    aux_source_directory("${path}/${pov}" pov_src)
    add_executable(${target} ${pov_src})

    # POVs are allowed to include challenge headers,
    # but must have their own definitions
    target_include_directories(${target} PUBLIC "${path}/${pov}" "${path}/include" "${path}/lib" "${path}/src")
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME "${pov}.pov"
        RUNTIME_OUTPUT_DIRECTORY "../pov"  # Put these with the other xml POVs
    )

    target_link_libraries(${target} LINK_PUBLIC cgc pov)
endfunction(buildPOV)


function(buildCB)
    # Generate all the patched #defines for this challenge
    set(patch_defs "-DPATCHED")
    if(VULN_COUNT)
        foreach(i RANGE 1 ${VULN_COUNT})
            set(patch_defs "${patch_defs} -DPATCHED_${i}")
        endforeach()
    endif()

    # Some challenges have multiple binaries that need to be built
    # Check if these directories exist
    file(GLOB cb_parts "${cb_path}/cb_*")
    if(cb_parts)
        # Iterate through the directories and build each
        set(i 1)
        foreach(cb ${cb_parts})
            build(${cb} ${cb_id}_${i})
            MATH(EXPR i "${i} + 1")
        endforeach()
    else()
        # Build normally if there are no extra directories
        build(${cb_path} ${cb_id})
    endif()

    file(GLOB cb_povs "${cb_path}/pov_*")
    if(cb_povs)
        foreach(path ${cb_povs})
            # Get the pov name, i.e. "pov_1"
            get_filename_component(pov ${path} NAME)
            buildPOV(${cb_path} ${cb_id} ${pov})
        endforeach()
    endif()
endfunction(buildCB)


# More options needed for cbs
add_compile_options(
    -nostdinc
    -fno-stack-protector
)

add_definitions(
    -Derrno=__cgc_errno
    -DFORTIFY_SOURCE=0
)

if(LINUX AND NOT CPIEVAL)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -z execstack -z norelro")
endif()

file(GLOB challenge_binaries processed-challenges/*)
foreach(cb_path ${challenge_binaries})
    if(IS_DIRECTORY ${cb_path} AND EXISTS ${cb_path}/CMakeLists.txt)
        # Get filename
        get_filename_component(cb_id ${cb_path} NAME)
        add_subdirectory(${cb_path} ${cb_path}/bin)
    endif()
endforeach()
