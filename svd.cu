#include "twoalg.h"

void tensor2maxtr(float *T,float *Ac,int a,int b,int c){ 
	float *T1 = new float[a*b*c];
	for(int i = 0;i<c;i++){
		for(int j = 0;j<a;j++){
			for(int t = 0;t<b;t++){
				T1[t*a*c+j*c+i] = T[i*a*b+j*b+t];   
			}
		}
	}                  //a*b*c变为a*c*b
	for(int i = 0;i<b;i++){
		for(int j = 0;j<a;j++){
			for(int k = 0;k<c;k++){
				T[i*a*c+k*a+j] = T1[i*a*c+j*c+k];
			}
			
		}
	}                 //a c转置一下 c*a*b
	for(int i = 0;i<b;i++){
		for(int j = 0;j<a*c;j++){
			T1[j*b+i] = T[i*a*c+j];    //T为按矩阵的行读写，再将矩阵转置 ，T1已经变成按行存储ac*b
		}
	}
	
	for(int i = 0;i<a*c;i++){
		
		for(int j = 0;j<b;j++){

			Ac[i*b*c+j] = T1[i*b+j];   //将原矩阵赋值进入Ac
				
		}
	}
	
	for(int k = 1;k<c;k++){
		for(int i = 0;i<a*c;i++){
			for(int j = 0;j<b;j++){
				Ac[i*b*c+k*b+j] = T1[((i+(c-k)*a)%(a*c))*b+j];
			}
		}            			//矩阵循环后赋值进入AC中
	}

}



void Msvd(float *A,float *U,float *S,float *V,int m,int n){   //实现矩阵的svd，A的大小为m*n
	cusolverDnHandle_t cusolverH = NULL;
	cudaStream_t stream = NULL;
	gesvdjInfo_t gesvdj_params = NULL;   //创建句柄	
	
	const int lda = m; //矩阵A的主维度
	//显存端分配空间
	
	float *d_A = NULL; /* device copy of A */
	float *d_S = NULL; /* singular values */
	float *d_U = NULL; /* left singular vectors */
	float *d_V = NULL; /* right singular vectors */
	int *d_info = NULL; /* error info */
	int lwork = 0;
	/* size of workspace */
	float *d_work = NULL; /* devie workspace for gesvdj */
	int info = 0;	/* host copy of error info */

	/* configuration of gesvdj */
	const double tol = 1.e-7;
	const int max_sweeps = 15;
	const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
	const int econ = 0 ; /* econ = 1 for economy size */
	/* numerical results of gesvdj*/
	
	cusolverDnCreate(&cusolverH);
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	cusolverDnSetStream(cusolverH, stream);
	cusolverDnCreateGesvdjInfo(&gesvdj_params);
	cusolverDnXgesvdjSetTolerance(
		gesvdj_params,
		tol);

	cusolverDnXgesvdjSetMaxSweeps(
		gesvdj_params,
		max_sweeps);
	cudaMalloc((void**)&d_A,sizeof(float)*lda*n);
	cudaMalloc((void**)&d_S,sizeof(float)*n);
	cudaMalloc((void**)&d_U,sizeof(float)*lda*m);
	cudaMalloc((void**)&d_V,sizeof(float)*n*n);
	cudaMalloc((void**)&d_info,sizeof(float));

	cudaMemcpy(d_A, A, sizeof(float)*lda*n,cudaMemcpyHostToDevice); //A传到GPU端
	
	cusolverDnSgesvdj_bufferSize(
		cusolverH,
		jobz, 	/* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
			/* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singularvectors */
		econ,    /* econ = 1 for economy size */
		m,    /* nubmer of rows of A, 0 <= m */
		n,   /* number of columns of A, 0 <= n */
		d_A,  /* m-by-n */
		lda,  /* leading dimension of A */
		d_S,  /* min(m,n) */
			/* the singular values in descending order */
		d_U,   /* m-by-m if econ = 0 */
			/* m-by-min(m,n) if econ = 1 */
		lda,    /* leading dimension of U, ldu >= max(1,m) */
		d_V,   /* n-by-n if econ = 0 */
			/* n-by-min(m,n) if econ = 1 */
		n,   	/* leading dimension of V, ldv >= max(1,n) */
		&lwork,
		gesvdj_params);

	cudaMalloc((void**)&d_work , sizeof(float)*lwork);

	cusolverDnSgesvdj(
	cusolverH,
		jobz, /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
			/* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singularvectors */
		econ, 	/* econ = 1 for economy size */
		m, 	/* nubmer of rows of A, 0 <= m */
		n,	/* number of columns of A, 0 <= n */
		d_A,	/* m-by-n */
		lda,	/* leading dimension of A */
		d_S,	/* min(m,n) */
			/* the singular values in descending order */
		d_U,
			/* m-by-m if econ = 0 */
			/* m-by-min(m,n) if econ = 1 */
		lda,
			/* leading dimension of U, ldu >= max(1,m) */
		d_V,
			/* n-by-n if econ = 0 */
			/* n-by-min(m,n) if econ = 1 */
		n,
			/* leading dimension of V, ldv >= max(1,n) */
		d_work,
		lwork,
		d_info,
		gesvdj_params);

	cudaDeviceSynchronize();

	cudaMemcpy(U, d_U, sizeof(float)*lda*m,cudaMemcpyDeviceToHost);
	cudaMemcpy(V, d_V, sizeof(float)*n*n,cudaMemcpyDeviceToHost);
	cudaMemcpy(S, d_S, sizeof(float)*n,cudaMemcpyDeviceToHost);
	cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	if ( 0 == info ){
	printf("gesvdj converges \n");
	}else if ( 0 > info ){
	printf("%d-th parameter is wrong \n", -info);
	exit(1);
	}else{
	printf("WARNING: info = %d : gesvdj does not converge \n", info );
	}
	printf("=====\n");

	cudaFree(d_A);
	cudaFree(d_S);
	cudaFree(d_U);
	cudaFree(d_V);
	cudaFree(d_info);
	cudaFree(d_work);
	
	cusolverDnDestroy(cusolverH);
	cudaStreamDestroy(stream);
	cusolverDnDestroyGesvdjInfo(gesvdj_params);
	cudaDeviceReset();

}





