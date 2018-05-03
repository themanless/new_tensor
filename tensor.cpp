#include <iostream>
class Tensor
{
    int m_m;
    int m_n;
    int m_k;
    double *m_array;

public:
    void setTensor(int m, int n, int k)
    {
        m_m = m;
        m_n = n;
        m_k = k;
        m_array = new double[m*n*k] {0};
    }

    void print()
    {
        for (int i = 0; i < m_k; i++)
        {
            for (int j = 0; j < m_m; j++)
            {
                for (int z = 0; z < m_n; z++)
                    std::cout << m_array[i*m_m*m_n + j*m_n + z] << " ";
                std::cout << "\n" ;
            }
            std::cout << "----------------------------------" << " \n";
        }
    }

};
