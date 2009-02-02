#ifndef CU_TIMER_H_
#define CU_TIMER_H_

void cuResetTimer();
float cuGetTimer(); // result in miliSec

#endif

/* use example : time launch & execution time for CUDA kernel
 
   cuResetTimer();  
   convoluteKernelA<<<  blkInGrid, thrdInBlk >>>(dA,dB);
   float tA=cuGetTimer();
   cudaThreadSynchronize(); 
   float tA2=cuGetTimer();
   printf("cuda kernel Lanunch   =%.1f usec, finished=%.1f msec\n",tA*1.e3,tA2);
*/
