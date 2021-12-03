
#include <chrono>
#include <random>

#include <KokkosBlas.hpp>
#include <KokkosSparse.hpp>

#include "LATypes.hpp"
#include "LoadMatrix.hpp"

template<class T>
Vect<T> rand_vect(size_t n, typename std::mt19937::result_type seed) {
	std::mt19937 engine(seed);
	// note floats are generated so that the values are the same regardless whether x is single or double
	std::uniform_real_distribution<float> dist;

	Vect<T> x ("Random vector", n);
	// Use regular loop so that the vector is deterministic for a given seed
	for (size_t i = 0; i<x.extent(0); i++) {
		x(i) = dist(engine);
	}
	return x;
}

template<class T>
MultiVect<T> rand_vect(size_t m, size_t n, typename std::mt19937::result_type seed) {
	std::mt19937 engine(seed);
	// note floats are generated so that the values are the same regardless whether x is single or double
	std::uniform_real_distribution<float> dist;

	MultiVect<T> x ("Random multivector", m, n);
	// Use regular loop so that the vector is deterministic for a given seed
	for (size_t i = 0; i<x.extent(0); i++) {
		for (size_t j = 0; j < x.extent(1); j++) {
			x(i, j) = dist(engine);
		}
	}
	return x;
}


template<class T>
int64_t test_dot(Vect<T> x, Vect<T> y) {

	auto t1 = std::chrono::high_resolution_clock::now();
	KokkosBlas::dot(x, y);
	auto t2 = std::chrono::high_resolution_clock::now();

	return std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

}


template<class T>
int64_t test_spmv(SparseMatrix<T> A, Vect<T> x, Vect<T> y) {

	auto t1 = std::chrono::high_resolution_clock::now();
	KokkosSparse::spmv("N", 1.0, A, x, 0.0, y);
	auto t2 = std::chrono::high_resolution_clock::now();

	return std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
}

template<class T>
int64_t test_gemv(MultiVect<T> V, Vect<T> x, Vect<T> y) {

	auto t1 = std::chrono::high_resolution_clock::now();
	KokkosBlas::gemv("T", 1.0, V, x, 0.0, y);
	auto t2 = std::chrono::high_resolution_clock::now();

	return std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
}

template<class T>
int64_t test_dot_axpy(Vect<T> x, Vect<T> y) {
	auto t1 = std::chrono::high_resolution_clock::now();
	T beta = KokkosBlas::dot(x, y);
	KokkosBlas::axpy(beta, x, y);
	auto t2 = std::chrono::high_resolution_clock::now();

	return std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
}

void evict(size_t eviction_bytes, char* eviction_array, char offset) {
	for (int i = 0; i < eviction_bytes; i += 64) {
		eviction_array[i+offset] = i;
	}
}

template<class T>
void run_trials(char* Apath, size_t v_cols, size_t eviction_bytes, int rand_seed) {
	const SparseMatrix<T> A = LoadMatrix<T>(Apath);
	const size_t n = A.numRows();

	Vect<T> x = rand_vect<T>(n, rand_seed);
	Vect<T> y = rand_vect<T>(n, rand_seed+1);
	Vect<T> z = rand_vect<T>(v_cols, rand_seed+2);
	MultiVect<T> V = rand_vect<T>(n, v_cols, rand_seed+3);

	char* eviction_array = new char[eviction_bytes];

	evict(eviction_bytes, eviction_array, 0);

	auto spmv_us = test_spmv(A, x, y);
	std::cout << "SPMV: " << spmv_us << " us" << std::endl;
	spmv_us = test_spmv(A, x, y);
	std::cout << "SPMV: " << spmv_us << " us" << std::endl;

	evict(eviction_bytes, eviction_array, 1);

	auto dot_us = test_dot(x, y);
	std::cout << "DOT:  " << dot_us << " us" << std::endl;
	dot_us = test_dot(x, y);
	std::cout << "DOT:  " << dot_us << " us" << std::endl;

	evict(eviction_bytes, eviction_array, 2);

	auto mgs_us = test_dot_axpy(x, y);
	std::cout << "MGS:  " << mgs_us << " us" << std::endl;
	mgs_us = test_dot_axpy(x, y);
	std::cout << "MGS:  " << mgs_us << " us" << std::endl;

	evict(eviction_bytes, eviction_array, 3);

	auto gemv_us = test_gemv(V, x, z);
	std::cout << "GEMV: " << gemv_us << " us" << std::endl;
	gemv_us = test_gemv(V, x, z);
	std::cout << "GEMV: " << gemv_us << " us" << std::endl;

	// ensure eviction_bytes is written to
	char sum = 0;
	for (int i = 0; i < eviction_bytes; i ++) {
		sum += eviction_array[i];
	}

	delete [] eviction_array;
}

int main(int argc, char* argv[]) {

	char* Apath = nullptr;
	size_t trials = 1;
	int rand_seed = 42;
	size_t v_cols = 100;
	size_t eviction_bytes = 1024*1024;

	for (int i = 1; i < argc; i++) {
		if (strcmp("--Apath", argv[i]) == 0) {
			Apath = argv[++i];
		} else if (strcmp("--trials", argv[i]) == 0) {
			trials = std::stoi(argv[++i]);
		} else if (strcmp("--vcols", argv[i]) == 0) {
			v_cols = std::stoi(argv[++i]);
		} else if (strcmp("--eviction", argv[i]) == 0) {
			eviction_bytes = std::stoll(argv[++i]);
		} else if (strcmp("--rand", argv[i]) == 0) {
			rand_seed = std::stoi(argv[++i]);
		} else {
			std::cout << "Unknown flag" << argv[i] << std::endl;
			return 1;
		}
	}
	if (Apath == nullptr) {
		std::cout << "No value suplied for A" << std::endl;
		return 1;
	}

	Kokkos::ScopeGuard kokkos(argc, argv);

	std::cout << "##### Warming MKL #####" << std::endl;
	run_trials<double>(Apath, v_cols, eviction_bytes, rand_seed);
	run_trials<float>(Apath, v_cols, eviction_bytes, rand_seed);

	std::cout << "##### Performance #####" << std::endl;
	std::cout << "### Double ###" << std::endl;
	run_trials<double>(Apath, v_cols, eviction_bytes, rand_seed);

	std::cout << "### Float ###" << std::endl;
	run_trials<float>(Apath, v_cols, eviction_bytes, rand_seed);
}

