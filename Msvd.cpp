#include "Msvd.h"

void Msvd(cufftComplex *A, cufftComplex *U, cufftComplex *V, float *S, int m, int n, cudaStream_t *streams, int i){   //实现矩阵的svd，A的大小为m*n
	cusolverDnHandle_t cusolverH = NULL;   //创建句柄	返回的是Ｖ的转置　按列存储的
    cusolverDnCreate(&cusolverH);
    cudaStreamCreate(&streams[i]);
    cusolverDnSetStream(cusolverH, streams[i]);
	
	const int lda = m; //矩阵A的主维度
	//显存端分配空间
	
	cufftComplex *d_A = NULL; /* device copy of A */
	float *d_S = NULL; /* singular values */
	cufftComplex *d_U = NULL; /* left singular vectors */
	cufftComplex *d_V = NULL; /* right singular vectors */
	int *devInfo = NULL;
	cufftComplex *d_work = NULL;
	float *d_rwork = NULL;

	int lwork = 0;
	int info_gpu = 0;
	
	cudaMalloc((void**)&d_A,sizeof(cufftComplex)*lda*n);
	cudaMalloc((void**)&d_S,sizeof(float)*n);
	cudaMalloc((void**)&d_U,sizeof(cufftComplex)*lda*m);
	cudaMalloc((void**)&d_V,sizeof(cufftComplex)*n*n);
	cudaMalloc((void**)&devInfo,sizeof(int));

	cudaMemcpy(d_A, A, sizeof(cufftComplex)*lda*n,cudaMemcpyHostToDevice); //A传到GPU端
	cusolverDnCgesvd_bufferSize(
		cusolverH,
		m,
		n,
		&lwork );
	
	cudaMalloc((void**)&d_work , sizeof(cufftComplex)*lwork);

	
	signed char jobu = 'A'; // all m columns of U
	signed char jobvt = 'A'; // all n columns of VT
	cusolverDnCgesvd (
		cusolverH,
		jobu,
		jobvt,
		m,
		n,
		d_A,
		lda,
		d_S,
		d_U,
		lda, // ldu
		d_V,
		n, // ldvt,
		d_work,
		lwork,
		d_rwork,
		devInfo);

	cudaDeviceSynchronize();

	cudaMemcpy(U, d_U, sizeof(cufftComplex)*lda*m,cudaMemcpyDeviceToHost);
	cudaMemcpy(V, d_V, sizeof(cufftComplex)*n*n,cudaMemcpyDeviceToHost);
	cudaMemcpy(S, d_S, sizeof(float)*n,cudaMemcpyDeviceToHost);
	cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	

	//printf("after gesvd: info_gpu = %d\n", info_gpu);

	//printf("=====\n");

	cudaFree(d_A);
	cudaFree(d_S);
	cudaFree(d_U);
	cudaFree(d_V);
	cudaFree(devInfo);
	cudaFree(d_work);
	cudaFree(d_rwork);
	
	cusolverDnDestroy(cusolverH);
	cudaDeviceReset();

}
