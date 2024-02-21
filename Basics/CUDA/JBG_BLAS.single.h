/////////////////////////////   General Purpose  FUNCTIONS
int Read_Matrix_file(float *matrix, int size, char *filename);
void Write_A_over_B(float *matrixA, float *matrixB, int m, int n);
void InverseMatrix(float *matrixA, int n);
void PrintColMatrix(float *matrix, int m, int n);
void MakeZeroes(float *matrix, int m, int n);
void MakeIdentity(float *matrix, int m, int n);
int MaxRow(float *matrix, int n);
int MaxRow2(float *matrix, int N, int n);
float MatrixDistance(float *matrixa, float *matrixb, int m, int n);
void SwapRow_ColMajMatrix(float *matrix, int from, int towards, int m, int n);
float float_machine_epsilon();

/////////////////////////////   Triangular Solver FUNCTIONS
void InitializeSubmatrices(float *matrixc, float *C11, float *C12, float *C21, float *C22, int m, int p);
void CollectSubmatrices(float *matrixc, float *C11, float *C12, float *C21, float *C22, int m, int p);
void UpperTriangularSolverRecursiveReal_0(float *matrixU, float *matrixB, float *matrixX, int n, int p);
void LowerTriangularSolverRecursiveReal_0(float *matrixL, float *matrixB, float *matrixX, int n, int p);
void LowerTriangularSolverNaiveReal(float *matrixL, float *matrixB, float *matrixSol, int n);
void UpperTriangularSolverNaiveReal(float *matrixU, float *matrixB, float *matrixSol, int n);

/////////////////////////////   LU Decomposition FUNCTIONS
void ipiv_to_P(int *ipiv, int n, float *P);
void SchurComplement(float *matrix, int N, int n);
void SchurComplement2(float *matrix, int n);
void LUdecompositionRecursive2(float *matrix, float *Lmatrix, float *Umatrix, int N, int n);
void LUdecompositionRecursive4Pivot(float *AmatrixBIG, float *LmatrixBIG, float *UmatrixBIG, int *IPIV, int N, int n);

/////////////////////////////// Error Calculating Functions
float InfinityNorm(float *matrixA, int n);
float InfinityNormVector(float *vectorA, int n);
float ConditionNumber(float *matrixA, int m, int n);
float residual_matrix(float *matrixA, float *matrixX, float *matrixB, float *matrixResidual, int m, int n);
void ErrorCalc_Display(float *matrixA, float *matrixB, float *matrixX, long double  elapsed_time, int n, int p);
void ErrorCalc_Display_v2(int i, float *matrixA, float *matrixB, float *matrixX, float *results, int m, int n);