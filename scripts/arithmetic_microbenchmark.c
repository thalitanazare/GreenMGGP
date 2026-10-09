/* Independent binary64 arithmetic microbenchmark. Compile without fast-math.
   Dependent chains measure loop + arithmetic latency, not application energy.
   Eight separate operations per iteration amortise loop overhead. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static volatile double sink;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }
__attribute__((noinline)) static double add(long count, double x, double a) {
 for(long i=0;i<count;i++){ x=x+a;x=x+a;x=x+a;x=x+a;x=x+a;x=x+a;x=x+a;x=x+a; } return x;
}
__attribute__((noinline)) static double mul(long count, double x, double a) {
 for(long i=0;i<count;i++){ x=x*a;x=x*a;x=x*a;x=x*a;x=x*a;x=x*a;x=x*a;x=x*a; } return x;
}
int main(void){
 const long n=10000000; puts("round,operation,ns_per_operation,result");
 for(int r=0;r<9;r++) for(int p=0;p<2;p++){
  int op=(r+p)%2; double t=now(); double result=op?mul(n,1.,1.000000000000001):add(n,1.,1e-12); sink=result;
  double ns=(now()-t)*1e9/(8.*n); printf("%d,%s,%.9f,%.17g\n",r,op?"multiply":"add",ns,result);
 }return 0;
}
