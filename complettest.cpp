#include "head.h"
#include "Tensor.h"
#include "CTensor.h"
#include "common.h"
#include "one_step.h"
#include <iostream>

int main(int argc, char **argv)
{
    int m = argv[1];
    int n = argv[2];
    int k = argv[3];
    int rank = argv[4];
    Tensor *t1 = new Tensor(m, rank, k, 'r');
    Tensor *t2 = new Tensor(rank, n, k, 'r');
    Tensor *T = new Tensor(*t1.Tprod(*t2));
    delete t1;
    t1 = nullptr;
    delete t2;
    t2 = nullptr;

    Tensor *omega = new Tensor(m, n, k);
    Tensor *T_omega = new Tensor(m, n, k);
    for (int i=0; i<m; i++)
        for (int j=0; j<n; j++)
            for (int z=0; z<k; z++)
                if (random(1000) < 0.5)
                {
                    *omega(i, j, z) = 1.0;
                    *T_omega(i, j, z) = T(i, j, z);
                }

    CTensor *T_omega_f = *T_omega.Tfft();
    delete T_omega;
    T_omega = nullptr;
    CTensor *omega_f = *omega.Tfft();
    delete omega;
    omega = nullptr;
    Tensor *Y = new Tensor(rank, n, k, 'r');
    CTensor *Y_f = *Y.Tfft();
    CTensor *Y_f_trans = new Tensor(n, rank, k);
    *Y_f_trans = *Y_f.Trans();
    CTensor *omega_f_k = new CTensor(m, n, k);
    for(int i=0; i<m; i++)
        for(int j=0; j<n; j++)
            for (int z=0; z<k; z++)
            {
                *omega_f_k(i, j, k).x = *omega_f(i, j, k).x / k;
                *omega_f_k(i, j, k).y = *omega_f(i, j, k).y / k;
            }
    CTensor *omega_f_trans = new CTensor(n, m, k);
    *omega_f_trans = *omega_f.Trans();
    CTensor *omega_f_trans_k = new CTensor(n, m, k);

    for(int i=0; i<n; i++)
        for(int j=0; j<m; j++)
            for (int z=0; z<k; z++)
            {
                *omega_f_trans_k(i, j, k).x = *omega_f_trans(i, j, k).x / k;
                *omega_f_trans_k(i, j, k).y = *omega_f_trans(i, j, k).y / k;
            }
    CTensor *T_omega_f_trans = new CTensor(n, m, k);
    *T_omega_f_trans = *T_omega_f.Trans();
    delete omega_f;
    omega_f = nullptr;
    delete omega_f_trans;
    omega_f_trans = nullptr;
    CTensor *X_f = new CTensor(m, rank, k);
    CTensor *X_f_trans = new CTensor(rank, m, k);
    for (int iter=0; iter<10; iter++)
    {
        *X_f_trans = one_step(T_omega_f_trans, omega_f_trans, Y_f_trans);
        *X_f = *X_f_trans.Trans();
        *Y_f = one_step(T_omega_f, omega_f, X_f);
        *Y_f_trans = *Y_f.Trans();
    }

 }
