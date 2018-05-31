#include <iostream>
#include "tensor.h"
#include "CTensor.h"

int main()
{
    int m = 3;
    int n = 2;
    int k = 2;
    Tensor ts(3, 2, 2);
    ts(1,1,0) = 2.0;
    ts(2,1,0) = 2.0;

    ts(0,1,0) = 2.0;
    ts(1,1,1) = 1.0;
    //cudaDeviceReset();

    Tensor U(m, m, k);
    Tensor S(m, n, k);
    Tensor V(n, n, k);

    std::cout << ts << "\n" ;
    ts.Tsvd(U, S, V);
    Tensor US(m, n, k);
    U.Tprod(S, US);
    Tensor USV(m, n, k);
    US.Tprod(V, USV);
    //Tensor US = U.Tprod(S);
    //Tensor USV = US.Tprod(V);
    std::cout << U << "\n U";
    std::cout << S << "\n S";
    std::cout << V << "\n V";
    std::cout << USV << "\n USV";
    //cudaDeviceReset();
    return 0;
}
