enable_language(ASM)

file(GLOB_RECURSE SRCS *.cpp *.c)

set(SRCS "${SRCS}")

if(ENABLE_SIMPLESTOT_ASM OR ENABLE_MR_KYBER)
  add_library(libOTe STATIC ${SRCS})
else()
  add_library(libOTe ${SRCS})
endif()


# make projects that include libOTe use this as an include folder
target_include_directories(libOTe PUBLIC
                    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/..>
                    $<INSTALL_INTERFACE:include>)
target_include_directories(libOTe PUBLIC
                    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
                    $<INSTALL_INTERFACE:include>)


target_link_libraries(libOTe cryptoTools)

if(MSVC)
    #target_compile_options(libOTe PRIVATE -openmp:experimental)
else()

    if(ENABLE_SSE)
        target_compile_options(libOTe PRIVATE -msse2)
    endif()
    if(ENABLE_AVX)
        target_compile_options(libOTe PRIVATE -maes -msse3 -msse4.1 -mpclmul -mavx2)
    endif()
endif()
if (ENABLE_BITPOLYMUL)
	target_link_libraries(libOTe bitpolymul)
endif ()
if(ENABLE_SIMPLESTOT_ASM)
	target_link_libraries(libOTe SimplestOT)
    #target_compile_options(libOTe PRIVATE -fPIC -no-pie)
endif()

if(ENABLE_MR_KYBER)
	target_link_libraries(libOTe KyberOT)
endif()



#############################################
#            Install                        #
#############################################

# install library
#install(TARGETS libOTe DESTINATION lib)

# install headers
#install(DIRECTORY . DESTINATION include/libOTe FILES_MATCHING PATTERN "*.h")
