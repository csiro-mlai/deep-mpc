#include "fe.h"

#include <stdio.h>

void fe_print(fe f) {
    unsigned char a[32];
    fe_tobytes(a, f);
    printf("\t");
    for(int i = 31; i >= 0; i--) {
        printf("%.2X", a[i]);
    }
    printf("\n");
}
