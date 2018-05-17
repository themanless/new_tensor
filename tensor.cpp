#include "tensor.h"
#include <iostream>
Tensor::Tensor(int m, int n, int k)
{
    m_m = m;
    m_n = n;
    m_k = k;
    m_array = new double[m*n*k] {0};
}

Tensor::Tensor(int m, int n, int k, char flag)
{
    m_m = m;
    m_n = n;
    m_k = k;
    m_array = new double[m*n*k] {0};
    if (flag == 'r')
    {
        for (int i=0; i<m*n*k; i++)
            m_array[i] = random(1000);
    }
    else if(flag == 'i')
    {
        for (int i=0; i<m*n; i++)
            m_array[i] = 1.0;
    }
    else if(flag == 'e')
    {
        assert(n == 1 && m == 1);
        m_array[0] = 1.0;
    }

}
std::ostream& operator<< (std::ostream &out, const Tensor &tensor)
{

    for (int i = 0; i < tensor.m_k; i++)
    {
        for (int j = 0; j < tensor.m_m; j++)
        {
            for (int z = 0; z < tensor.m_n; z++)
                out << tensor.m_array[i*tensor.m_m*tensor.m_n + j*tensor.m_n + z] << " ";
            out << "\n" ;
        }
        out << "----------------------------------" << " \n";
    }
    out << "***************END***************" << "\n";
}

double& Tensor::operator()(int m, int n, int k)
{
    return m_array[k*m_m*m_n + m*m_n + n];
}
const double& Tensor::operator()(int m, int n, int k) const
{
    return m_array[k*m_m*m_n + m*m_n + n];
}
bool operator==(Tensor &t1, Tensor &t2)
{
    bool flag = true;
    int i = 0;
    while (i < t1.m_m * t1.m_n * t.m_k)
    {
        if (abs(t1.m_array[i] - t2.m_array[i]) <= 1E-4)
        {
            flag = false;
            break;
        }
    }
    return flag;
}
Tensor Tensor::getlaslice(int n)
{
    Tensor slice(m_m, 1, m_k);
    for (int i=0; i<m_k; i++)
        for (int j=0; j<m_m; j++)
            slice(j, 0, i) = *this(j,n,i);
    return slice;
}
CTensor Tensor::Tfft()
{
    int m = m_m;
    int n = m_n;
    int l = m_k;
    int bat = m*n;
    cufftComplex *t_f = new cufftComplex[l*bat];
    for(int i=0;i<bat;i++)
      for(int j=0;j<l;j++){
        t_f[i*l+j].x=m_array[j*bat+i];
        t_f[i*l+j].y=0;
      }
    cufftComplex *d_fftData;
    CHECK(cudaMalloc((void**)&d_fftData,l*bat*sizeof(cufftComplex)));

    CHECK(cudaMemcpy(d_fftData,t_f,l*bat*sizeof(cufftComplex),cudaMemcpyHostToDevice));

    cufftHandle plan =0;
    CHECK_CUFFT(cufftPlan1d(&plan,l,CUFFT_C2C,bat));
    CHECK_CUFFT(cufftExecC2C(plan,(cufftComplex*)d_fftData,(cufftComplex*)d_fftData,CUFFT_FORWARD));
    cudaDeviceSynchronize();
    CHECK(cudaMemcpy(t_f,d_fftData,l*bat*sizeof(cufftComplex),cudaMemcpyDeviceToHost));

    cufftDestroy(plan);
    cudaFree(d_fftData);
//transform
    CTensor tf(m, n, l);
    for(int i=0; i<m; i++)
        for(int j=0; j<n; j++)
            for (int k=0; k<l; k++)
            {
                tf(i, j, k) = t_f[(i*n+j)*l +k];
            }
    delete[] t_f;
    t_f = nullptr;
    return tf;

}
Tensor Tensor::Trans()
{
    Tensor t(m_n, m_m, m_k);
    for (int i; i<m_k; i++)
        for (int j; j<m_m; j++)
            for (int z; z<m_n; z++)
                t(z, j, i) = *this(j, z, i);
    return t;
}
Tensor Tensor::Tprod(Tensor &t2)
{
    int row = m_m;
    int rank = m_n;
    int tupe = m_k;
    int col = t2.getN();
    CTensor t1f = this->Tfft();
    CTensor t2f = t2.Tfft();
    CTensor Tf(row, col, tupe);
    //mul of Tensor
    for(int i=0; i<tupe;i++){
        for(int j=0;j<row;j++){
          for(int k=0;k<col;k++){
            for(int w=0;w<rank;w++){
              mul_cufft(t1f(j, w, i), t2f(w, k, i), Tf(j, k, i));
        }
      }
    }
  }
    return (Tf.Tifft());
}
double Tensor::norm2()
{
    double c = 0;
    for (int i=0; i<m_n*m_m*m_k; i++)
        c += m_array[i]*m_array[i];
    c = sqrt(c);
    return c;
}
Tensor Tensor::Tinnpro(Tensor &t2)
{
    assert(m_n == 1 && t2.getN() == 1);
    return (*this.Tprod(t2));
}
bool Tensor::Tort()
{
    Tensor zero(1, 1, m_k);
    bool flag = true;
    for (int i=0; i<m_n; i++)
        for (int j=i; j<m_n;j++)
        {
            if (j==i)
            {
                Tensor slice(*this.getlaslice(j));
                Tensor iiinn = slice.Tprod(slice);
                for (int z=1; z<m_k; z++)
                {
                    if(abs(iiinn(0, 0, z)) > 1E-4)
                    {
                        flag = false ;
                        return flag;
                    }
                }
            }
            else
            {
                Tensor slicej(*this.getlaslice(j));
                Tensor slicei(*this.getlaslice(i));
                Tensor ijinn = slicej.Tprod(slicei);
                for (int z=0; z<m_k; z++)
                {
                    if(abs(ijinn(0, 0, z)) > 1E-4)
                    {
                        flag = false ;
                        return flag;
                    }
                }

            }
        }
    return flag;
}
void Tensor::Tsvd(Tensor &U, Tensor &S, Tensor &V)
{

}
Tensor Tensor::Tinv()
{
    Tensor U(m_m, m_m, m_k);
    Tensor S(m_m, m_n, m_k);
    Tensor V(m_n, m_n, m_k);
    Tsvd(U, S, V);
    invs(S);
    Tensor US = U.Tprod(S);
    return (US.Tprod(V));

}
