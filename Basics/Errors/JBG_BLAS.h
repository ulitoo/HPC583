/////////////////////////////   General Purpose  FUNCTIONS
int Read_Matrix_file(double *matrix, int size, char *filename);
void Write_A_over_B(double *matrixA, double *matrixB, int m, int n);
void InverseMatrix(double *matrixA, int n);
void PrintColMatrix(double *matrix, int m, int n);
void MakeZeroes(double *matrix, int m, int n);
void MakeIdentity(double *matrix, int m, int n);
int MaxRow(double *matrix, int n);
int MaxRow2(double *matrix, int N, int n);
double MatrixDistance(double *matrixa, double *matrixb, int m, int n);
void SwapRow_ColMajMatrix(double *matrix, int from, int towards, int m, int n);
double machine_epsilon();

/////////////////////////////   Triangular Solver FUNCTIONS
void InitializeSubmatrices(double *matrixc, double *C11, double *C12, double *C21, double *C22, int m, int p);
void CollectSubmatrices(double *matrixc, double *C11, double *C12, double *C21, double *C22, int m, int p);
void UpperTriangularSolverRecursiveReal_0(double *matrixU, double *matrixB, double *matrixX, int n, int p);
void LowerTriangularSolverRecursiveReal_0(double *matrixL, double *matrixB, double *matrixX, int n, int p);
void LowerTriangularSolverNaiveReal(double *matrixL, double *matrixB, double *matrixSol, int n);
void UpperTriangularSolverNaiveReal(double *matrixU, double *matrixB, double *matrixSol, int n);

/////////////////////////////   LU Decomposition FUNCTIONS
void ipiv_to_P(int *ipiv, int n, double *P);
void SchurComplement(double *matrix, int N, int n);
void SchurComplement2(double *matrix, int n);
void LUdecompositionRecursive2(double *matrix, double *Lmatrix, double *Umatrix, int N, int n);
void LUdecompositionRecursive4Pivot(double *AmatrixBIG, double *LmatrixBIG, double *UmatrixBIG, int *IPIV, int N, int n);

/////////////////////////////// Error Calculating Functions
double InfinityNorm(double *matrixA, int n);
double InfinityNormVector(double *vectorA, int n);
double ConditionNumber(double *matrixA, int m, int n);
void ErrorCalc_Display(double *matrixA, double *matrixB, double *matrixX, long double elapsed_time, int n, int p);
void ErrorCalc_Display_v2(double *matrixA, double *matrixB, double *matrixX, long double elapsed_time, int n, int p);