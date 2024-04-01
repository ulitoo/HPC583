


void init_sum(float *a, float r,int n);
void init_sum2(float *a, float r,int n);
float finitesum_CPU(float *a, int n);
float finitesum_exact(float r, int n);
float kahanSum(const float* array, size_t size);
__global__ void finitesum_GPU(float *a, float *c, int size);
__global__ void reduce0(float *g_idata, float *g_odata);
__global__ void reduce1(float *g_idata, float *g_odata);
__global__ void reduce2(float *g_idata, float *g_odata);

