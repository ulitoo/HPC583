#ifndef SCALAPACK_H
#define SCALAPACK_H

#include <complex>
typedef std::complex<float> complex_s;
typedef std::complex<double> complex_d;
typedef struct
{
	int desctype;
	int ctxt;
	int m;
	int n;
	int nbrow;
	int nbcol;
	int sprow;
	int spcol;
	int lda;
} MDESC;

extern "C"
{
	void Cblacs_pinfo(int *mypnum, int *nprocs);
	void Cblacs_get(int context, int request, int *value);
	void Cblacs_gridinit(int *context, char *order, int np_row, int np_col);
	void Cblacs_gridinit(int *, char *, int, int);
	void Cblacs_gridinfo(int context, int *np_row, int *np_col, int *my_row, int *my_col);
	void Cblacs_gridexit(int context);
	void Cblacs_exit(int error_code);
	void Cblacs_gridmap(int *context, int *map, int ld_usermap, int np_row, int np_col);

	int npreroc_(int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);
	int numroc_(int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);

	void descinit_(MDESC idescal, int *m, int *n, int *mb, int *nb, int *dummy1, int *dummy2, int *icon, int *procRows, int *info);

	void psgesvd_(char *jobu, char *jobvt, int *m, int *n, float *a, int *ia, int *ja, int *desca, float *s, float *u, int *iu, int *ju, int *descu, float *vt, int *ivt, int *jvt, int *descvt, float *work, int *lwork, int *info);
	void pdgesvd_(char *jobu, char *jobvt, int *m, int *n, double *a, int *ia, int *ja, int *desca, double *s, double *u, int *iu, int *ju, int *descu, double *vt, int *ivt, int *jvt, int *descvt, double *work, int *lwork, int *info);
	void pcgesvd_(char *jobu, char *jobvt, int *m, int *n, complex_s *a, int *ia, int *ja, int *desca, float *s, complex_s *u, int *iu, int *ju, int *descu, complex_s *vt, int *ivt, int *jvt, int *descvt, complex_s *work, int *lwork, float *rwork, int *info);
	void pzgesvd_(char *jobu, char *jobvt, int *m, int *n, complex_d *a, int *ia, int *ja, int *desca, double *s, complex_d *u, int *iu, int *ju, int *descu, complex_d *vt, int *ivt, int *jvt, int *descvt, complex_d *work, int *lwork, double *rwork, int *info);
	void pdgemm_(char *jobu, char *jobvt, int *, int *, int *, double *, double *, int *, int *, int *, double *, int *, int *, int *, double *, double *, int *, int *, int *);
	void pdgeadd_(char TRANS, int *M, int *N, double *ALPHA, double *A, int *IA, int *JA, int *DESCA, double *BETA, double *C, int *IC, int *JC, int *DESCC);
	void PDGEMM(char TRANSA, char TRANSB, int M, int N, int K, double ALPHA, double *A, int IA, int JA, int *DESCA,
				double *B, int IB, int JB, int *DESCB, double BETA, double *C, int IC, int JC, int *DESCC);
	void PDGESV(int N, int NRHS, double *A, int IA, int JA, int *DESCA, int *IPIV, double *B, int IB, int JB, int *DESCB, int &INFO);

	void Cpdgemr2d(int m, int n, double *ptrmyblock, int ia, int ja, MDESC *ma, double *ptrmynewblock, int ib, int jb, MDESC *mb, int globcontext);
	//void Cblacs_pcoord(int, int, int*, int*)
}

#endif
