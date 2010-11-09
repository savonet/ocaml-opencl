__kernel void Convolve(const __global float *in,
                        __constant float *filter,
                        __global float *out,
                        const int in_width,
                        const int filter_width)
{
    const int width = get_global_size(0);

    const int xOut = get_global_id(0);
    const int yOut = get_global_id(1);

    const int xInTopLeft = xOut;
    const int yInTopLeft = yOut;

    float sum = 0;
    for (int r = 0; r < filter_width; r++)
    {
        const int idxFtmp = r * filter_width;

        const int yIn = yInTopLeft + r;
        const int idxIntmp = yIn * in_width + xInTopLeft;

        for (int c = 0; c < filter_width; c++)
        {
            const int idxF  = idxFtmp  + c;
            const int idxIn = idxIntmp + c;
            sum += filter[idxF]*in[idxIn];
        }
    }
    const int idxOut = yOut * width + xOut;
    out[idxOut] = sum;
}
