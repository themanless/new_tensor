cc = nvcc
pro = test
src = test.cpp tensor.cpp CTensor.cpp 
flag = -std=c++11 -lcufft

$(pro):$(src)
	$(cc) $(flag) -o $(pro) $(src)
