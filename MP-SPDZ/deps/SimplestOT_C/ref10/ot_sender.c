#include "ot_sender.h"

#include <stdlib.h>

#include "ge.h"


void ref10_sender_genS(ref10_SENDER * s, unsigned char * S_pack)
{
	int i;

	ge_p3 S, yS;

	//

	sc_random(s->y, 0);

	ge_scalarmult_base(&S, s->y); // S

	ge_p3_tobytes(S_pack, &S); // E^0(S)

	for (i = 0; i < 3; i++) ge_p3_dbl_p3(&S, &S); // 8S

	ge_p3_tobytes(s->S_pack, &S); // E_1(S)

	ge_scalarmult_vartime(&yS, s->y, &S);
	for (i = 0; i < 3; i++) ge_p3_dbl_p3(&yS, &yS); // 64T
	s->yS = yS;
}

void ref10_sender_keygen(ref10_SENDER * s,
                   unsigned char * Rs_pack,
                   unsigned char (*keys)[4][HASHBYTES])
{
	for (int i = 0; i < 4; i++)
		sender_keygen_part(s, &Rs_pack[i * HASHBYTES], keys, i);
}

void sender_keygen_part(ref10_SENDER * s,
                   unsigned char * Rs_pack, 
                   unsigned char (*keys)[4][HASHBYTES],
				   int j)
{
	int i;

	ge_p3 P0;
	ge_p3 P1;
	ge_p3 Rs;
	ge_cached tmp;
	ge_p1p1 tmp2;

	//

	if (ge_frombytes_vartime(&Rs, Rs_pack) != 0)
	{ 
		fprintf(stderr, "Error: point decompression failed\n"); exit(-1);
	}

	for (i = 0; i < 3; i++) ge_p3_dbl_p3(&Rs, &Rs); // 64R^i

	ge_p3_tobytes(Rs_pack, &Rs); // E_2(R^i)

	ge_scalarmult_vartime(&P0, s->y, &Rs); // 64yR^i
	ge_hash(keys[0][j], s->S_pack, Rs_pack, &P0); // E_2(yR^i)

	ge_p3_to_cached(&tmp, &P0);
	ge_sub(&tmp2, &s->yS, &tmp); // 64(T-yR^i)
	ge_p1p1_to_p3(&P1, &tmp2);
	ge_hash(keys[1][j], s->S_pack, Rs_pack, &P1); // E_2(T - yR^i)
}

