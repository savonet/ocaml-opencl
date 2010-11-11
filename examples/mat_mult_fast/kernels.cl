#define BLOCK_SIZE 32

//#pragma OPENCL EXTENSION cl_amd_printf:enable

__kernel
__attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void Mat_mult(__global float *a,
                       __global float *b,
                       __global float *c,
                       int n,
                       int p)
{
  int i = get_group_id(0);
  int j = get_group_id(1);
  int ii = get_local_id(0);
  int jj = get_local_id(1);
  int k, kk;
  float sum = 0;

  for (k = 0; k < n / BLOCK_SIZE; k++)
    {
      __local float aa[BLOCK_SIZE][BLOCK_SIZE];
      __local float bb[BLOCK_SIZE][BLOCK_SIZE];

      aa[ii][jj] = a[(i*BLOCK_SIZE+ii)*n+(k*BLOCK_SIZE+jj)];
      bb[ii][jj] = b[(k*BLOCK_SIZE+ii)*p+(j*BLOCK_SIZE+jj)];

      barrier(CLK_LOCAL_MEM_FENCE);

      for (kk = 0; kk < BLOCK_SIZE; kk++)
        sum += aa[ii][kk] * bb[kk][jj];

      barrier(CLK_LOCAL_MEM_FENCE);
    }
  c[(i*BLOCK_SIZE+ii)*p+(j*BLOCK_SIZE+jj)] = sum;
}
