__kernel void Convolve(const __global float *in,
                        __constant float *filter,
                        __global float *out,
                        const int in_width,
                        const int filter_width)
{
    const int width = get_global_size(0);

    const int x = get_global_id(0);
    const int y = get_global_id(1);

    float sum = 0;
    for (int r = 0; r < filter_width; r++)
        for (int c = 0; c < filter_width; c++)
            sum += filter[r * filter_width + c] * in[y * in_width + x + c];
    out[y * width + x] = sum;
}
