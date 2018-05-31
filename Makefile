cc = nvcc
pro = test
src = test.cpp tensor.cpp CTensor.cpp Msvd.cpp
flag = -std=c++11 -lcufft -lcusolver -lcusparse

$(pro):$(src)
	$(cc) $(flag) -o $(pro) $(src)
