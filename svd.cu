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

void Mfsvd(cufftComplex *A,cufftComplex *U,float *S,cufftComplex *V,int m,int n){   //实现矩阵的svd，A的大小为m*n
	cusolverDnHandle_t cusolverH = NULL;   //创建句柄	返回的是Ｖ的转置　按列存储的
	
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
	cusolverDnCreate(&cusolverH);
	
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

void cuinverse(cufftComplex *A,cufftComplex *A_f,int m){  //A 为原矩阵，A_f为逆矩阵
	cufftComplex *U = new cufftComplex[m*m];  //存放左特征向量
	cufftComplex *V = new cufftComplex[m*m];	//存放右特征向量
	float *S = new float[m];
	cufftComplex *UT = new cufftComplex[m*m];
	cufftComplex *VT = new cufftComplex[m*m];

	Mfsvd(A,U,S,V,m,m);  //实现方矩阵的奇异值分解

	/*for(int i = 0;i<m;i++){
		cout<<S[i]<<" "<<endl;
	}
	cout<<endl;
	for(int i = 0;i<m;i++){
		for(int j = 0;j<m;j++){
			cout<<U[j*m+i].x<<"+"<<U[j*m+i].y<<"i"<<" ";
		}
		cout<<endl;
	}
	cout<<"_____+++____"<<endl;

	for(int i = 0;i<m;i++){
		for(int j = 0;j<m;j++){
			cout<<V[j*m+i].x<<"+"<<V[j*m+i].y<<"i"<<" ";
		}
		cout<<endl;
	}                        //V里存的是V的转置
	cout<<"_____"<<endl;
*/
	
	for(int i = 0;i<m;i++){
		S[i] = 1/S[i];
		//cout<<S[i]<<" "<<endl;
	}
	cout<<endl;      //S阵取逆就是取倒数

	for(int i = 0;i<m;i++){
		for(int j = 0;j<m;j++){
			VT[i*m+j].x= V[j*m+i].x;
			VT[i*m+j].y= 0-V[j*m+i].y;
			
		}
	}   
 	/*for(int i = 0;i<m;i++){
		for(int j = 0;j<m;j++){
			cout<<VT[j*m+i].x<<"+"<<VT[j*m+i].y<<"i"<<" ";
		}
	cout<<endl;
	}
	cout<<" +++"<<endl;
*/
	for(int i = 0;i<m;i++){
		for(int j = 0;j<m;j++){
			VT[i*m+j].x = (VT[i*m+j].x)*S[i];
			VT[i*m+j].y = (VT[i*m+j].y)*S[i];
		}
	}
	
	/* for(int i = 0;i<m;i++){
		for(int j = 0;j<m;j++){
			cout<<VT[j*m+i].x<<"+"<<VT[j*m+i].y<<"i"<<" ";
		}
	cout<<endl;
	}
*/
	 for(int i = 0;i<m;i++){
		for(int j = 0;j<m;j++){
			UT[i*m+j].x = U[j*m+i].x;  //U的逆就是转置
			UT[i*m+j].y = 0 - U[j*m+i].y;
		}
		
	}
	

	/*for(int i = 0;i<m;i++){
		for(int j = 0;j<m;j++){
			cout<<UT[j*m+i].x<<"+"<<UT[j*m+i].y<<"i"<<" ";
		}
		cout<<endl;
	}                        //U的转置UT
	cout<<"_____"<<endl;
	*/

	for(int i = 0;i<m;i++){
		for(int j = 0;j<m;j++){
			U[j*m+i] = UT[i*m+j];
			
		}
	}          //将UT按行存储

	for(int i = 0;i<m;i++){
		for(int j = 0;j<m;j++){
			V[j*m+i] = VT[i*m+j];
			
		}  //将VT按行存储
	} 

	/*for(int i = 0;i<m;i++){
		for(int j = 0;j<m;j++){
			cout<<V[i*m+j].x<<"+"<<V[i*m+j].y<<"i"<<" ";
		}
		cout<<endl;
	}                        //V
	cout<<"_____"<<endl;
	for(int i = 0;i<m;i++){
		for(int j = 0;j<m;j++){
			cout<<U[i*m+j].x<<"+"<<U[i*m+j].y<<"i"<<" ";
		}
		cout<<endl;
	}                        //U
	cout<<"_____"<<endl;
*/
	//printfTensor(m,m,1,V);
	//printfTensor(m,m,1,U);
	
	mul_pro(V,U,A_f,m,m,m);//A_f为逆矩阵
	//printfTensor(m,m,1,A_f);

	
}

float psnr(float *image1,float *image2,int m,int n,int k){

	float PSNR = 0.0;
	float MSE = 0.0;

	for(int i = 0;i<m*n*k;i++){
		image1[i] = image1[i]*255;
		image2[i] = image2[i]*255;
	}
	
	for(int j = 0;j<k;j++){
		for(int a = 0;a<m*n;a++){
			MSE = MSE+((image1[a] - image2[a])*(image1[a] - image2[a]));		
		}	
		MSE = MSE/m*n;
		PSNR = PSNR+10*log10(255*255/MSE);

	}
	PSNR = PSNR/k;
	return PSNR;

}












