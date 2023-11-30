#include "ge.h"

/*
r = 2 * p
*/

void ge_p3_dbl(ge_p1p1 *r,const ge_p3 *p)
{
  ge_p2 q;
  ge_p3_to_p2(&q,p);
  ge_p2_dbl(r,&q);
}

void ge_p3_dbl_p3(ge_p3 *s,const ge_p3 *p)
{
  ge_p2 q;
  ge_p1p1 r;
  ge_p3_to_p2(&q,p);
  ge_p2_dbl(&r,&q);
  ge_p1p1_to_p3(s, &r);
}
