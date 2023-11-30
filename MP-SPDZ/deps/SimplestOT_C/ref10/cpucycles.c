#include <stdio.h>
#include <sys/types.h>

long long cpucycles_amd64cpuinfo(void)
{
#ifdef __x86_64__
  unsigned long long result;
  asm volatile(".byte 15;.byte 49;shlq $32,%%rdx;orq %%rdx,%%rax"
    : "=a" (result) ::  "%rdx");
  return result;
#else
  return 0;
#endif
}
