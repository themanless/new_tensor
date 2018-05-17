#include "head.h"
#include <fstream>
#include "fft.h"
#include <cufft.h>
#include <iostream>
#include "cuda_runtime.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include "tprod.h"
#include "time.h"
#include "twoalg.h"
#define patsize 5;
#define basenum 30;  //基的数量

using namespace std;

/*void printMatrix(int m, int n, const float*A, int lda){
		for(int row = 0 ; row < m ; row++){
			for(int col = 0 ; col < n ; col++){
				double Areg = A[row + col*lda];
				printf("(%d,%d) = %20.16E\n",  row+1, col+1,Areg);
			}
		}
	}*/

int main(int argc,char *argv[]){

	const int a = atoi(argv[1]);
	const int b = atoi(argv[2]);  //a b为矩阵的大小
	float *A = new float[a*b];
	const int lda = a;
	srand(2);
	for(int i = 0;i<a;i++){
		for(int j = 0;j<b;j++){
			A[j*a+i] = rand()%5+1;
		}
	}           //随机生成一个矩阵A，大小为a*b
	
	for(int i = 0;i<a;i++){
		for(int j = 0;j<b;j++){
			cout<<A[j*a+i]<<" ";
		}
		cout<<endl;
	}
	cout<<"_________________"<<endl;
	//调用函数
	float *U = new float[a*a];  //存放左特征向量
	float *V = new float[b*b];	//存放右特征向量
	float *S = new float[b];
	Msvd(A,U,S,V,a,b);
	cout<<S[0]<<endl;
	for(int i = 0;i<b;i++){
		cout<<S[i]<<" "<<endl;
	}
	cout<<endl;
	for(int i = 0;i<a;i++){
		for(int j = 0;j<a;j++){
			cout<<U[j*a+i]<<" ";
		}
		cout<<endl;
	}
	cout<<"_____"<<endl;

	for(int i = 0;i<b;i++){
		for(int j = 0;j<b;j++){
			cout<<V[j*b+i]<<" ";
		}
		cout<<endl;
	}
	cout<<"_____"<<endl;

}


/*

	int a = atoi(argv[1]);
	int b = atoi(argv[2]);
	int c = atoi(argv[3]);   //确定张量各个维度的大小
	float *T = new float[a*b*c];
	float *T1 = new float[a*b*c];
	ifstream read(argv[4]);
	for(int i = 0;i<a*b*c;i++){
	read>>T[i];
	}
	for(int i = 0;i<c;i++)
		for(int j = 0;j<a;j++)
			for(int k = 0;k<b;k++){
				T1[i*a*b+j*b+k] = T[j*b*c+i*b+k];
			}

	srand(2);
	for(int i = 0;i<a*b*c;i++){
		T[i] = rand()%4+1;  //随机生成张量中的数据0~10,为Nmsi(加了噪声的脏数据)
	}
	
	for(int i = 0;i<c;i++){
		for(int j = 0;j<b;j++){
			for(int k = 0;k<a;k++){
				cout<<T[i*a*b+j*a+k]<<" ";
			}
			cout<<endl;
		}
	cout<<"_____"<<endl;
	}
	
	
	for(int i = 0;i<a*b*c;i++){
		T1[i] = rand()%4+1;  //随机生成张量中的数据0~10,为Nmsi(加了噪声的脏数据)
	}

	for(int i = 0;i<a*b*c;i++){
		cout<<T[i]<<" ";
		//cout<<T1[i]<<" ";
	}
	cout<<endl;
	cout<<endl;
	printTensor(a,b,c,T);
	printTensor(a,b,c,T1);


	//检测傅里叶变换
	clock_t start,finish;
	start = clock();
	cufftComplex *tf = new cufftComplex[a*b*c];
	Tfft(T,c,a*b,tf);
	printfTensor(a,b,c,tf);   //初步成功
	//检测共轭转置
	cufftComplex *temp  = new cufftComplex[a*b*c];
	Tftranspose(tf,temp,a,b,c);
	printfTensor(a,b,c,temp); //初步成功

	//检测逆傅里叶变换
	float *TT = new float[a*b*c];
	Tifft(TT,c,a*b,tf);
        
	printTensor(a,b,c,TT);
        delete[] TT;
	finish = clock();
	
	cout<<(double)(finish-start)/CLOCKS_PER_SEC;
	cout<<endl;


   
    //检测张量的转置
	
	float *temp = new float[a*b*c];
	Ttranspose(TT,temp,a,b,c);
	printTensor(a,b,c,temp);     //初步成功

	//检测张量积
	float *result = new float[a*b*c];
	for(int i = 0;i<a*b*c;i++){
		result[i] = 0;  
	}

	tprod(T,T1,result,a,b,b,c);
	printTensor(a,b,c,result);
	delete[] result;

*/




