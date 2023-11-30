#ifndef REF10_OT_SENDER_H
#define REF10_OT_SENDER_H

#include <stdio.h>

#include "ge.h"
#include "sc.h"
#include "ot_config.h"

typedef struct
{
	unsigned char S_pack[ PACKBYTES ];
	unsigned char y [32];
	ge_p3 yS;
} ref10_SENDER;

void ref10_sender_genS(ref10_SENDER *, unsigned char *);
void ref10_sender_keygen(ref10_SENDER *, unsigned char *, unsigned char (*)[4][HASHBYTES]);

void sender_keygen_part(ref10_SENDER *, unsigned char *, unsigned char (*)[4][HASHBYTES], int);

#endif //ifndef REF10_OT_SENDER_H

