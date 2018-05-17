#include <iostream>
#include "tensor.h"
#include "CTensor.h"

int main()
{
    Tensor ts(3, 2, 2);
    ts(1,1,0) = 2.0;
    ts(2,1,0) = 2.0;

    ts(0,1,0) = 2.0;
    ts(1,1,1) = 1.0;
    cudaDeviceReset();
    Tensor t2(2, 3, 2);
    t2(1,1,0) = 2.0;
    t2(1,2,0) = 2.0;

    t2(0,1,0) = 2.0;
    t2(1,1,1) = 1.0;

    Tensor t(ts.Tprod(t2));
    std::cout << ts << "\n" << t2 << "\n";
    std::cout << t << "\n";
    cudaDeviceReset();
    return 0;
}
