cmake_minimum_required(VERSION 3.12)
project(SimplestOT C ASM)

set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -Wextra -no-pie -fno-PIC")

include_directories(.)

set(SRCS_ASM
    consts.s
    consts4x.s
    cpucycles.c
    cpucycles.h
    crypto_hash.h
    fe25519.h
    fe25519_add.c
    fe25519_freeze.s
    fe25519_getparity.c
    fe25519_invert.c
    fe25519_iseq_vartime.c
    fe25519_mul.s
    fe25519_neg.c
    fe25519_nsquare.s
    fe25519_pack.c
    fe25519_pow2523.c
    fe25519_setint.c
    fe25519_square.s
    fe25519_sub.c
    fe25519_unpack.c
    ge25519.h
    ge25519_add.c
    ge25519_add_p1p1.s
    ge25519_dbl_p1p1.s
    ge25519_double.c
    ge25519_lookup.s
    ge25519_lookup_niels.s
    ge25519_nielsadd2.s
    ge25519_p1p1_to_p2.s
    ge25519_p1p1_to_p3.s
    ge25519_pack.c
    ge25519_scalarmult.c
    ge25519_scalarmult_base.c
    ge25519_setneutral.c
    ge25519_unpack.c
    ge4x.c
    ge4x.data
    ge4x.h
    ge4x_add_p1p1.s
    ge4x_double_p1p1.s
    ge4x_lookup.s
    ge4x_lookup_niels.s
    ge4x_niels_add_p1p1.s
    ge4x_pack.c
    ge4x_unpack_vartime.c
    gfe4x.c
    gfe4x.h
    gfe4x_add.s
    gfe4x_getparity.c
    gfe4x_iseq_vartime.c
    gfe4x_mul.s
    gfe4x_nsquare.c
    gfe4x_pow2523.c
    gfe4x_square.s
    gfe4x_sub.s
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
    sc25519.h
    sc25519_from32bytes.c
    sc25519_random.c
    sc25519_window4.c
    to_4x.h
    )
add_library(SimplestOT_ASM STATIC ${SRCS_ASM})

add_executable(ot_sender_asm ../common/ot_sender_test.c)
target_link_libraries(ot_sender_asm SimplestOT_ASM)

add_executable(ot_receiver_asm ../common/ot_receiver_test.c)
target_link_libraries(ot_receiver_asm SimplestOT_ASM)
