#include <amp.h>  
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
using namespace concurrency;

// one side of the matrix
#define L 256
// the width of the matrix
#define A L*L

// matrix multiply function with concurrency 
__int64 mat_mul(array_view<int, 2> v1, array_view<int, 2> v2, array_view<int, 2> tmp) {
	// using chrono library for checking [ms] time
	chrono::system_clock::time_point begin = chrono::system_clock::now();
	// using parallel_for_each to using concurrency
	parallel_for_each(
		tmp.extent,
		// lambda expression for kernel parameter
		[=](index<2> idx) restrict(amp) {
		int row = idx[0];
		int col = idx[1];
		for (int inner = 0; inner < L; inner++) {
			tmp[idx] += v1(row, inner) * v2(inner, col);
		}
	}
	);
	// synchronizing process like linux join function
	tmp.synchronize();
	chrono::duration<double> sec = chrono::system_clock::now() - begin;
	chrono::milliseconds mil = chrono::duration_cast<chrono::milliseconds>(sec);
	// stub code for test right value
	/*for (int row = 0; row < L; row++) {
		for (int col = 0; col < L; col++) {
			std::cout << tmp(row, col) << "  ";
		}
		std::cout << "\n";
	}*/
	// return [ms]
	return mil.count();
}

int main() {

	vector<int> v1(A, 1);
	vector<int> v2(A, 1);
	vector<int> tmp(A);
	
	// for concurrency calculation
	// divide vector array -> two dimension matrix L x L;
	array_view<int, 2> av1(L, L, v1);
	array_view<int, 2> av2(L, L, v2);
	array_view<int, 2> atmp(L, L, tmp);
	cout.precision(4);
	__int64 time = mat_mul(av1, av2, atmp);
	cout << "CPU Cocurrency Program" << endl;
	cout << L << " x " << L << " Matrix Multiply time > " << time << endl;
}
