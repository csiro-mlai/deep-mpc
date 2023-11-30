#include <stdio.h>
#include "ge.h"
#include "crypto_hash.h"

void ge_hash(unsigned char * k,
               unsigned char * sp,
               unsigned char * q,
               ge_p3 * p)
{
    int j;

    unsigned char r[32];
    unsigned char in[96];

    //

    ge_p3_tobytes(r, p);

    for (j = 0; j < 32; j++) in[j] = sp[j];

    for (j = 0; j < 32; j++) in[j + 32] = q[j];
    for (j = 0; j < 32; j++) in[j + 64] = r[j];

    crypto_hash(k, in, sizeof(in));
}

void ge_p3_cmov(ge_p3* out, ge_p3* in, unsigned char b) {
    fe_cmov(out->X, in->X, b);
    fe_cmov(out->Y, in->Y, b);
    fe_cmov(out->Z, in->Z, b);
    fe_cmov(out->T, in->T, b);
}

void ge_p3_print(ge_p3 *p, const char* info) {
    printf("%s:\n", info);
    fe_print(p->X);
    fe_print(p->Y);
    fe_print(p->Z);
    fe_print(p->T);
}
