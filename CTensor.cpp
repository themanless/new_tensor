#include "CTensor.h"

cufftComplex zero{0.0, 0.0};
CTensor::CTensor(int m, int n, int k)
{
    m_m = m;
    m_n = n;
    m_k = k;
    m_array = new cufftComplex[m*n*k] {zero};
}
std::ostream& operator<< (std::ostream &out, const CTensor &tensor)
{

    for (int i = 0; i < tensor.m_k; i++)
    {
        for (int j = 0; j < tensor.m_m; j++)
        {
            for (int z = 0; z < tensor.m_n; z++)
                out << tensor.m_array[i*tensor.m_m*tensor.m_n + j*tensor.m_n + z].x << "+" << tensor.m_array[i*tensor.m_m*tensor.m_n + j*tensor.m_n + z].y << "i ";
            out << "\n" ;
        }
        out << "----------------------------------" << " \n";
    }
    out << "***************END***************" << "\n";
}
cufftComplex& CTensor::operator()(int m, int n, int k)
{
    return m_array[k*m_m*m_n + m*m_n + n];
}
const cufftComplex& CTensor::operator()(int m, int n, int k) const
{
    return m_array[k*m_m*m_n + m*m_n + n];
}
Tensor CTensor::Tifft()
{
    int m = m_m;
    int n = m_n;
    int l = m_k;
    int bat = m*n;
    cufftComplex *t_f = new cufftComplex[l*bat];
//transform
    for(int i=0;i<bat;i++)
      for(int j=0;j<l;j++){
        t_f[i*l+j]=m_array[j*bat+i];
      }
    cufftComplex *d_fftData;
    cudaMalloc((void**)&d_fftData,l*bat*sizeof(cufftComplex));
    cudaMemcpy(d_fftData,t_f,l*bat*sizeof(cufftComplex),cudaMemcpyHostToDevice);

    cufftHandle plan =0;
    cufftPlan1d(&plan,l,CUFFT_C2C,bat);
    cufftExecC2C(plan,(cufftComplex*)d_fftData,(cufftComplex*)d_fftData,CUFFT_INVERSE);
    cudaDeviceSynchronize();
    cudaMemcpy(t_f,d_fftData,l*bat*sizeof(cufftComplex),cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_fftData);
//transform
    Tensor t(m, n, l);
    for(int i=0; i<m; i++)
        for(int j=0; j<n; j++)
            for (int k=0; k<l; k++)
            {
                t(i, j, k) = t_f[(i*n+j)*l +k].x/l;
            }
    delete[] t_f;
    t_f = nullptr;
    return t;
}
CTensor CTensor::Trans()
{
CTensor t(m_n, m_m, m_k);
    for (int i; i<m_k; i++)
        for (int j; j<m_m; j++)
            for (int z; z<m_n; z++)
            {
                t(z, j, i).x = *this(j, z, i).x;
                t(z, j, i).y = 0.0 - *this(j, z, i).y;
            }

}
void mul_cufft(cufftComplex &a,cufftComplex &b,cufftComplex &c){
  c.x += a.x*b.x - a.y*b.y;
  c.y += a.x*b.y + a.y*b.x;
}

