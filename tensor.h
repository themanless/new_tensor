#ifndef TENSOR_H
#define TENSOR_H
#include <iostream>
#include "CTensor.h"
#include <assert.h>
#include <math.h>
class CTensor;

class Tensor
{
    int m_m;
    int m_n;
    int m_k;
    double *m_array;

public:
    Tensor(int m, int n, int k);

    Tensor(int m, int n, int k, char flag);

    friend std::ostream& operator<< (std::ostream &out, const Tensor &tensor);

    int getM() {return m_m;}
    int getN() {return m_n;}
    int getK() {return m_k;}
    double& operator()(int m, int n, int k);
    const double& operator()(int m, int n, int k) const;
    friend bool& operator==(Tensor &t1, Tensor &t2);
    Tensor getlaslice(int n);
    CTensor Tfft();
    Tensor Tprod(Tensor &t2);
    double norm2();
    Tensor Tinnpro(Tensor &t2);
    bool Tort(Tensor &t2);
    void Tsvd(Tensor U, Tensor S, Tensor V);
    Tensor Tinv();
    ~Tensor()
    {
        delete[] m_array;
        m_array = nullptr;
    }

};

#endif
