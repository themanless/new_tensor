#include <iostream>
#include "Msvd.h"
#include <cufft.h>
#include <cusolverDn.h>
void mul_cufft(cufftComplex *a,cufftComplex *b,cufftComplex *c){
  c->x += a->x*b->x - a->y*b->y;
  c->y += a->x*b->y + a->y*b->x;
}
void mulmat(cufftComplex *a, cufftComplex *b, cufftComplex *c, int m, int n, int k)
{
    for (int i=0; i<m; i++)
        for (int j=0; j<k; j++)
            for (int z=0; z<n; z++)
            {
                mul_cufft(a+z*m+i, b+j*n+z, c+j*m+i);
            }

}
int main()
{
    int m = 3;
    int n = 2;
    cufftComplex A[m*n];
    for (int i=0; i<6; i++)
    {
        A[i].x = i;
        A[i].y = i;

    }
    cufftComplex U[m*m];
    cufftComplex V[n*n];
    float S[n];
    Msvd(A, U, V, S, m, n);
    cufftComplex SS[n*n];
    for (auto &item: SS)
    {
        item.x =0.0;
        item.y =0.0;
    }
    for (int i=0; i<n; i++)
    {
        SS[i*n+i].x = S[i];
    }
    cufftComplex US[m*n];
    for (auto &item: US)
    {
        item.x =0.0;
        item.y =0.0;
    }
    mulmat(U, SS, US, m, n, n);
    cufftComplex USV[m*n];
    for (auto &item: USV)
    {
        item.x =0.0;
        item.y =0.0;
    }
    cufftComplex VT[n*n];
    for (int i =0; i<n; i++)
        for (int j =0; j<n; j++)
        {
            VT[i*n+j].x = V[j*n+i].x;
            VT[i*n+j].y = 0.0-V[j*n+i].y;
        }
    mulmat(US, V, USV, m, n, n);
    for (auto &item : A)
        std::cout << item.x << " " << item.y << "i ";
    std::cout << "A\n";
    for (auto &item : U)
        std::cout << item.x << " " << item.y << "i ";
    std::cout << "U\n";
    for (auto &item : V)
        std::cout << item.x << " " << item.y << "i ";
    std::cout << "V\n";
    for (auto &item : S)
        std::cout << item << " ";
    std::cout << "S\n";
    for (auto &item : USV)
        std::cout << item.x << " " << item.y << "i ";
    std::cout << "USV\n";
    return 0;
}
