#ifndef OT_RECEIVER_H
#define OT_RECEIVER_H

#include <stdio.h>

#include "sc25519.h"
#include "ge4x.h"
#include "ot_config.h"

struct ot_receiver
{
	unsigned char S_pack[ PACKBYTES ];
	ge4x S;
	ge4x table[ 64/DIST ][8];

	// temporary

	ge4x xB;
	sc25519 x[4];
};

typedef struct ot_receiver RECEIVER;

void receiver_maketable(RECEIVER *);
void receiver_procS(RECEIVER *);
void receiver_rsgen(RECEIVER *, unsigned char *, unsigned char *);
void receiver_keygen(RECEIVER *, unsigned char (*)[HASHBYTES]);

#endif //ifndef OT_RECEIVER_H

