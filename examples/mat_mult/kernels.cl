__kernel void Mat_mult(__global float *a,
                       __global float *b,
                       __global float *c,
                        int n,
                        int p)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k;
    float sum = 0;

    for (k = 0; k < n; k++)
      sum += a[i * n + k] * b[k * p + j];
    c[i * p + j] = sum;
}
