#ifndef REF10_OT_RECEIVER_H
#define REF10_OT_RECEIVER_H

#include <stdio.h>

#include "sc.h"
#include "ge.h"
#include "ot_config.h"

typedef struct
{
	unsigned char S_pack[ PACKBYTES ];
	ge_p3 S;

	// temporary

	ge_p3 xB[4];
	unsigned char x[4][32];
} ref10_RECEIVER;

void receiver_rsgen_part(ref10_RECEIVER *, unsigned char *, unsigned char, int i);
void receiver_keygen_part(ref10_RECEIVER *, unsigned char [HASHBYTES], int i);

// compatibility
void ref10_receiver_maketable(ref10_RECEIVER *);
void ref10_receiver_procS(ref10_RECEIVER *);
void ref10_receiver_rsgen(ref10_RECEIVER *, unsigned char *, unsigned char *);
void ref10_receiver_keygen(ref10_RECEIVER *, unsigned char (*)[HASHBYTES]);

#endif //ifndef REF10_OT_RECEIVER_H

