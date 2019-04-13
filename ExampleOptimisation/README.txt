
To compile all the performances tests, run them and do the plots :
	./compileTest.sh

To do the same step by step :

mkdir build
cd build
cmake ..
make
make run_all
make plot_all

If you want to change the compiler :

mkdir build
cd build
cmake .. -DCMAKE_C_COMPILER=/path/to/C_compiler -DCMAKE_CXX_COMPILER=/path/to/C++_compiler 
make
make run_all
make plot_all

