

__global__ void finitesum_GPU(float *a, float *c, int size);
void init_sum(float *a, float r,int n);
float finitesum_CPU(float *a, int n);
float finitesum_exact(float r, int n);
