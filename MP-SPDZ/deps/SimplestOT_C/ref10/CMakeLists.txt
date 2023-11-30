cmake_minimum_required(VERSION 3.12)
project(SimplestOT C)

set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -Wextra -DREF10 -std=gnu99")

include_directories(.)

set(SRCS
    crypto_hash.h
    cpucycles.c
    fe.h
    fe.c
    fe_0.c
    fe_1.c
    fe_add.c
    fe_cmov.c
    fe_copy.c
    fe_frombytes.c
    fe_invert.c
    fe_isnegative.c
    fe_isnonzero.c
    fe_mul.c
    fe_neg.c
    fe_pow22523.c
    fe_sq.c
    fe_sq2.c
    fe_sub.c
    fe_tobytes.c
    ge.h
    ge.c
    ge_add.c
    ge_add.h
    ge_double_scalarmult.c
    ge_frombytes.c
    ge_madd.c
    ge_madd.h
    ge_msub.c
    ge_msub.h
    ge_p1p1_to_p2.c
    ge_p1p1_to_p3.c
    ge_p2_0.c
    ge_p2_dbl.c
    ge_p2_dbl.h
    ge_p2_dbl.q
    ge_p3_0.c
    ge_p3_dbl.c
    ge_p3_to_cached.c
    ge_p3_to_p2.c
    ge_p3_tobytes.c
    ge_precomp_0.c
    ge_scalarmult_base.c
    ge_sub.c
    ge_sub.h
    ge_tobytes.c
    Keccak-simple-settings.h
    Keccak-simple.c
    network.c
    network.h
    ot_config.h
    ot_receiver.c
    ot_receiver.h
    ot_sender.c
    ot_sender.h
    ../common/randombytes.c
    ../common/randombytes.h
    sc.h
    sc_muladd.c
    sc_random.c
    sc_reduce.c
    verify.c
    )
add_library(SimplestOT STATIC ${SRCS})

add_executable(ot_sender ../common/ot_sender_test.c)
target_link_libraries(ot_sender SimplestOT)

add_executable(ot_receiver ../common/ot_receiver_test.c)
target_link_libraries(ot_receiver SimplestOT)
