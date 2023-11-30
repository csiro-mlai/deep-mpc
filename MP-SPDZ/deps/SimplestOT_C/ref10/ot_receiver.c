#include "ot_receiver.h"

#include <stdlib.h>

#include "ge.h"

void ref10_receiver_maketable(ref10_RECEIVER * r)
{
	(void) r;
}

void ref10_receiver_procS(ref10_RECEIVER * r)
{
	int i;

	ge_p3 S;

	if (ge_frombytes_vartime(&S, r->S_pack) != 0)
	{ 
		fprintf(stderr, "Error: point decompression failed\n"); exit(-1);
	}

	for (i = 0; i < 3; i++) ge_p3_dbl_p3(&S, &S); // 8S

	ge_p3_tobytes(r->S_pack, &S); // E_1(S)
	r->S = S;
}

void receiver_rsgen_part(ref10_RECEIVER * r,
                     unsigned char * Rs_pack,
                     unsigned char c,
					 int i)
{
	ge_p1p1 P;
	ge_p3 P_tmp;
	ge_cached xB;
	
	sc_random(r->x[i], 1);
	ge_scalarmult_base(&r->xB[i], r->x[i]); // 8x^iB

	ge_p3_to_cached(&xB, &r->xB[i]);

	ge_sub(&P, &r->S, &xB); // 8S - 8x^iB
	ge_p1p1_to_p3(&P_tmp, &P);
	ge_p3_cmov(&r->xB[i], &P_tmp, c);

	ge_p3_tobytes(Rs_pack, &r->xB[i]); // E^1(R^i)

}

void ref10_receiver_rsgen(ref10_RECEIVER* r, unsigned char* Rs_pack, unsigned char* cs)
{
	for (int i = 0; i < 4; i++)
		receiver_rsgen_part(r, &Rs_pack[i * PACKBYTES], cs[i], i);
}

void receiver_keygen_part(ref10_RECEIVER * r,
                     unsigned char keys[HASHBYTES],
					 int j)
{
	int i;

	unsigned char Rs_pack[ PACKBYTES ];
	ge_p3 P;
	
	//

	for (i = 0; i < 3; i++) ge_p3_dbl_p3(&r->xB[j], &r->xB[j]);
	ge_p3_tobytes(Rs_pack, &r->xB[j]); // E_2(R^i)

	ge_scalarmult_vartime(&P, r->x[j], &r->S); // 64x^iS

	ge_hash(keys, r->S_pack, Rs_pack, &P); // E_2(x^iS)
}

void ref10_receiver_keygen(ref10_RECEIVER * r, unsigned char (*keys)[HASHBYTES])
{
	for (int i = 0; i < 4; i++)
		receiver_keygen_part(r, keys[i], i);
}
