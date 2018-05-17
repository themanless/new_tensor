#include<stdio.h>
#include<stdlib.h>
#include<cusolverSp.h>
#include<cusparse_v2.h>
#include<cuda_runtime.h>
#include<cublas_v2.h>
/**
 * @ complex tensor: a,b,c
 * Create on:May 9 2018
 * @author:da xu
 *
 */
void t(const int m,const int n,const int k,cuComplex* Aarray[],
		cuComplex* Barray[],cuComplex* Carray[],int batchCount){
	cublasOperation_t transa=CUBLAS_OP_N;
	cublasOperation_t transb=CUBLAS_OP_N;
	cuComplex alpha;
	alpha.x=1;
	alpha.y=0;
	cuComplex beta;
	beta.x=0;
	beta.y=0;
	int lda=m;
	int ldb=k;
	int ldc=m;
    cuComplex *d_Aarray[batchCount];
    cuComplex *d_Barray[batchCount];
	cuComplex *d_Carray[batchCount];
	for(int i=0;i<batchCount;i++){
	cudaMalloc((void**)&d_Aarray[i],sizeof(cuComplex)*m*k);
	cudaMalloc((void**)&d_Barray[i],sizeof(cuComplex)*k*n);
	cudaMalloc((void**)&d_Carray[i],sizeof(cuComplex)*m*n);
	cudaMemcpy(d_Aarray[i],Aarray[i],sizeof(cuComplex)*m*k,cudaMemcpyHostToDevice);
	cudaMemcpy(d_Barray[i],Barray[i],sizeof(cuComplex)*k*n,cudaMemcpyHostToDevice);
	}
    const cuComplex **d_A;
    const cuComplex **d_B;
    cuComplex **d_C;
    cudaMalloc((void**)&d_A, sizeof(cuComplex *)*batchCount);
    cudaMalloc((void**)&d_B, sizeof(cuComplex *)*batchCount);
    cudaMalloc((void**)&d_C, sizeof(cuComplex *)*batchCount);
	cudaMemcpy(d_A, d_Aarray, sizeof(cuComplex *)*batchCount, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, d_Barray, sizeof(cuComplex *)*batchCount, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, d_Carray, sizeof(cuComplex *)*batchCount, cudaMemcpyHostToDevice);
	//const cuComplex* a[batchCount]={d_Aarray[0],d_Aarray[1],d_Aarray[2],d_Aarray[3]};
	cublasStatus_t stat;
	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	stat=cublasCgemmBatched(handle,
			transa,
			transb,
			m,n,k,
			&alpha,
			d_A,lda,
			d_B,ldb,
			&beta,
			d_C,ldc,
			batchCount);
	if(stat==CUBLAS_STATUS_SUCCESS){
		printf("success\n");
	}
	for(int i=0;i<batchCount;i++){
	cudaMemcpy(Carray[i],d_Carray[i],sizeof(cuComplex)*m*n,cudaMemcpyDeviceToHost);
	}
	cublasDestroy_v2(handle);
    for (int i=0; i<batchCount; i++)
    {
        cudaFree(d_Aarray[i]);
        cudaFree(d_Barray[i]);
        cudaFree(d_Carray[i]);
    }
    cudaFree(d_C);
    cudaFree(d_A);
    cudaFree(d_B);


}
int main(int arc ,char** argv){
	/*const int m=4;
	const int n=2;
	const int nnz=4;
	printf("kds");
	int row[5]={0,1,2,3,4};
	int col[4]={0,0,1,1};
	double A[4]={1.0,2.0,3.0,4.0};
	cusparseHandle_t handle1;
	cusparseCreate(&handle1);
	cusparseMatDescr_t descrA;
	cusparseCreateMatDescr(&descrA);
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
	double* b=(double*)malloc(sizeof(double)*m);
	b[0]=1.0; b[1]=2.0;b[2]=4.0;b[3]=5.0;
	double tol=0.000001;
	int rank;
	double* x=(double*)malloc(sizeof(double)*n);
	int* p=(int*)malloc(sizeof(int)*4);
	double  min_norm;
	cusolverSpHandle_t handle;
	cusolverSpCreate(&handle);
	cusolverSpDcsrlsqvqrHost(handle,m,n,nnz,descrA,
			A,row,col,b,
			0.000001,&rank,x,p,&min_norm);
	for(int i=0;i<n;i++){
		printf("X[%d]=%f\n",i,x[i]);
	}*/

	int m=2;int n=2; int k=2;int batchCount=4;
	cuComplex *Aarray[batchCount];
	cuComplex* Barray[batchCount];
	cuComplex* Carray[batchCount];
	for(int i=0;i<batchCount;i++){
    Aarray[i]=(cuComplex*)malloc(sizeof(cuComplex)*m*k);
    Barray[i]=(cuComplex*)malloc(sizeof(cuComplex)*k*n);
    Carray[i]=(cuComplex*)malloc(sizeof(cuComplex)*m*n*2);

	}
   for(int i=0;i<batchCount;i++){
	   for(int j=0;j<m*k;j++){
	   Aarray[i][j].x=j;
	   Aarray[i][j].y=0;}
   }
   for(int i=0;i<batchCount;i++){
   	   for(int j=0;j<n*k;j++){
   	   Barray[i][j].x=j+1;
   	   Barray[i][j].y=0;
      }
   }
   t(m,n,k,Aarray,Barray,Carray,batchCount);
for(int i=0;i<batchCount;i++){
	for(int j=0;j<m*n;j++){
	printf("%f %f       ",Carray[i][j].x,Carray[i][j].y);}

	printf("\n##########################################\n");

}
	return 0;
}
