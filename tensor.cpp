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
                out << tensor(j, z, i) << " ";
            out << "\n" ;
        }
        out << "----------------------------------" << " \n";
    }
    out << "***************END***************" << "\n";
}

double& Tensor::operator()(int m, int n, int k)
{
    return m_array[k*m_m*m_n + n*m_m + m];
}
const double& Tensor::operator()(int m, int n, int k) const
{
    return m_array[k*m_m*m_n + n*m_m + m];
}
bool& operator==(Tensor &t1, Tensor &t2)
{
    bool flag = true;
    int i = 0;
    while (i < t1.m_m * t1.m_n * t1.m_k)
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
            slice(j, 0, i) = (*this)(j,n,i);
    return slice;
}
void Tensor::Tfft(CTensor &tf)
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
    cufftComplex* tfA = tf.getArray();
     for(int i=0;i<bat;i++)
        for(int j=0;j<l;j++){
            tfA[j*bat+i]=t_f[i*l+j];
             }
    delete[] t_f;
    t_f = nullptr;
}
Tensor Tensor::Trans()
{
    Tensor t(m_n, m_m, m_k);
    for (int i; i<m_k; i++)
        for (int j; j<m_m; j++)
            for (int z; z<m_n; z++)
                t(z, j, i) = (*this)(j, z, i);
    return t;
}
void Tensor::Tprod(Tensor t2, Tensor &t)
{
    int row = m_m;
    int rank = m_n;
    int tupe = m_k;
    int col = t2.getN();
    //CTensor t1f = this->Tfft();
    CTensor t1f(m_m, m_n, m_k);
    this->Tfft(t1f);
    //CTensor t2f = t2.Tfft();
    CTensor t2f(t2.getM(), t2.getN(), t2.getK());
    t2.Tfft(t2f);
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
    Tf.Tifft(t);
}
double Tensor::norm2()
{
    double c = 0;
    for (int i=0; i<m_n*m_m*m_k; i++)
        c += m_array[i]*m_array[i];
    c = sqrt(c);
    return c;
}
void Tensor::Tinnpro(Tensor &t2, Tensor &t)
{
    assert(m_n == 1 && t2.getN() == 1);
    this->Tprod(t2, t);
}
/*bool Tensor::Tort(Tensor &t2)
{
    Tensor zero(1, 1, m_k);
    bool flag = true;
    for (int i=0; i<m_n; i++)
        for (int j=i; j<m_n;j++)
        {
            if (j==i)
            {
                Tensor slice(this->getlaslice(j));
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
                Tensor slicej(this->getlaslice(j));
                Tensor slicei(this->getlaslice(i));
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
}*/
void Tensor::Tsvd(Tensor &U, Tensor &S, Tensor &V)
{
    CTensor Tf(m_m, m_n, m_k);
    (*this).Tfft(Tf);
    std::cout << *this << "\n ts \n";
    std::cout << Tf << "\n Tf \n";
    std::cout << "======" << "\n";
    cufftComplex *tp = Tf.getArray();
    CTensor Uf(m_m, m_m, m_k);
    cufftComplex *up = Uf.getArray();
    CTensor Vf(m_n, m_n, m_k);
    cufftComplex *vp = Vf.getArray();
    CTensor Sf(m_m, m_n, m_k);
    std::cout << "======" << "\n";
    float vecs[m_n*m_k];
    cudaStream_t streams[m_k];
    for (int i=0; i<m_k; i++)
    {
        Msvd(tp+i*m_n*m_m, up+i*m_m*m_m, vp+i*m_n*m_n, vecs+i*m_n, m_m, m_n, streams, i);
    }
    for (int i=0; i<m_k; i++)
    {
        cudaStreamDestroy(streams[i]);
    }
    //copy vecs to sp
    for (int i=0; i<m_k; i++)
        for (int j=0; j<m_n; j++)
            Sf(j, j, i).x = vecs[i*m_n + j];
    std::cout << Uf << "\n Uf \n";
    std::cout << Sf << "\n Sf \n";
    std::cout << Vf << "\n Vf \n";
    Uf.Tifft(U);
    Sf.Tifft(S);
    std::cout << S << "\n S \n";
    Vf.Tifft(V);
}
//Tensor Tensor::Tinv()
//{
    //Tensor U(m_m, m_m, m_k);
    //Tensor S(m_m, m_n, m_k);
    //Tensor V(m_n, m_n, m_k);
    //Tsvd(U, S, V);
    //invs(S);
    //Tensor US = U.Tprod(S);
    //return (US.Tprod(V));

//}
