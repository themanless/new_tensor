#ifndef CTENSOR_H
#define CTENSOR_H
#include <iostream>
#include <cufft.h>
#include <cuda_runtime.h>
#include "header.h"
#include "common.h"
#include "tensor.h"
class Tensor;
class CTensor
{
    int m_m;
    int m_n;
    int m_k;
    cufftComplex *m_array;

public:
    CTensor(int m, int n, int k);

    friend std::ostream& operator<< (std::ostream &out, const CTensor &tensor);

    cufftComplex *getArray() {return m_array;}
    int getM() {return m_m;}
    int getN() {return m_n;}
    int getK() {return m_k;}
    cufftComplex& operator()(int m, int n, int k);
    const cufftComplex& operator()(int m, int n, int k) const;
    void Tifft(Tensor &t);
    CTensor Trans();
    ~CTensor()
    {
        delete[] m_array;
        m_array = nullptr;
    }

};
void mul_cufft(cufftComplex &a,cufftComplex &b,cufftComplex &c);
#endif
