
#include "kernels.hpp"
#include "types_mkl.hpp"
#include "types_cuda.hpp"

#include <chrono>
#include <random>

#include <Kokkos_Core.hpp>

#include "gmres.hpp"
#include "IterUtil.hpp"
#include "LoadMatrix.hpp"
#include "Orthogonalization.hpp"


enum orth_t {
	cgs,
	mgs,
	cgsr,
	householder
};

enum prec_t {
	ilu,
	ilu_jacobi,
	jacobi,
	identity
};

enum test_mode_t {
	mixed_mode,
	baseline_mode,
	single_prec_mode,
	single_mode,
};


Vect<double, MKL> rand_vect(int n, typename std::mt19937::result_type seed=0) {
	std::mt19937 engine(seed);
	// note floats are generated so that the values are the same regardless whether x is single or double
	std::uniform_real_distribution<float> dist;

	Vect<double, MKL> x (n);
	double* x_data = x.data();
	// Use regular loop so that the vector is deterministic for a given seed
	for (unsigned int i = 0; i<x.n(); i++) {
		x_data[i] = dist(engine);
	}
	return x;
}

template<class ORTH, class Device, class Type, class PrecType = double>
void DoBaselineProblem(Convergence<Type, Device>& convergence,
					   const SparseMatrix<double, Device> A,
					   const Vect<double, Device> b,
					   const Vect<double, Device> true_x,
					   const prec_t prec_type,
					   int jacobi_steps) {

	std::cout << "Doing Baseline test" << std::endl;

	const int n = A.nrows();

	auto prec_start = std::chrono::high_resolution_clock::now();
	const SparseMatrix<float, Device> A_type (A);

	LinearOperator<PrecType, Device>* M;
	switch(prec_type) {
	case ilu: {
		ILU<PrecType, Device> M_ilu = ilu0<PrecType, Device>(A);
		M = new ILU<PrecType, Device>(M_ilu);
		break;
	}
	case ilu_jacobi: {
		ILU<PrecType, Device> M_ilu = ilu0<PrecType, Device>(A);
		M = new ILU_Jacobi<PrecType, Device>(M_ilu, jacobi_steps);
		break;
	}
	case jacobi: {
		M = new Jacobi<PrecType, Device>(A);
		break;
	}
	case identity: {
		M = new Identity<PrecType, Device>();
		break;
	}
	default: {
		std::cerr << "Unknown prec type " << prec_type << std::endl;
		return;
	}
	}
	std::chrono::duration<float> prec_time = std::chrono::high_resolution_clock::now() - prec_start;

	Vect<Type, Device> x_type (n);
	fill(0.0, x_type);
	Vect<Type, Device> b_type (n);
	copy(b, b_type);

	auto gmres_start = std::chrono::high_resolution_clock::now();
	gmres_baseline<ORTH, Device, Type, PrecType>(convergence, A_type, M, b_type, x_type);
	std::chrono::duration<float> gmres_time = std::chrono::high_resolution_clock::now() - gmres_start;

	Vect<double, Device> x(n);
	copy(x_type, x);
	Vect<double, Device> r (n);
	copy(b_type, r);
	spmv(-1.0, A, x, 1.0, r);
	auto res_norm = nrm2(r);

	axpy(-1.0, true_x, x);
	auto err_norm = nrm2(x);

	std::cout << "  ilu took " << prec_time.count() << "s; gmres took " << gmres_time.count() << "s" << std::endl;
	std::cout << "  resNorm = " << res_norm << "; errNorm = " << err_norm << std::endl;

	delete M;
}

template<class ORTH, class Device>
void DoMixedPrecisionProblem(Convergence<float, Device>& convergence,
							 const SparseMatrix<double, Device> A,
							 const Vect<double, Device> b,
							 const Vect<double, Device> true_x,
							 const prec_t prec_type,
	  						 int jacobi_steps) {

	std::cout << "Doing Mixed Precision test" << std::endl;

	const int n = A.nrows();

	Vect<double, Device> x (n);
	fill(0.0, x);

	auto prec_start = std::chrono::high_resolution_clock::now();
	const SparseMatrix<float, Device> A_single (A);

	LinearOperator<float, Device>* M;
	switch(prec_type) {
	case ilu: {
		ILU<float, Device> M_ilu = ilu0<float, Device>(A);
		M = new ILU<float, Device>(M_ilu);
		break;
	}
	case ilu_jacobi: {
		ILU<float, Device> M_ilu = ilu0<float, Device>(A);
		M = new ILU_Jacobi<float, Device>(M_ilu, jacobi_steps);
		break;
	}
	case jacobi: {
		M = new Jacobi<float, Device>(A);
		break;
	}
	case identity: {
		M = new Identity<float, Device>();
		break;
	}
	default: {
		std::cerr << "Unknown prec type " << prec_type << std::endl;
		return;
	}
	}
	std::chrono::duration<float> prec_time = std::chrono::high_resolution_clock::now() - prec_start;

	auto gmres_start = std::chrono::high_resolution_clock::now();
	gmres_singleUpdate<ORTH, Device>(convergence, A, A_single, M, b, x);
	std::chrono::duration<float> gmres_time = std::chrono::high_resolution_clock::now() - gmres_start;

	Vect<double, Device> r (n);
	copy(b, r);
	spmv(-1.0, A, x, 1.0, r);
	auto res_norm = nrm2(r);

	axpy(-1.0, true_x, x);
	auto err_norm = nrm2(x);

	std::cout << "  ilu took " << prec_time.count() << "s; gmres took " << gmres_time.count() << "s" << std::endl;
	std::cout << "  resNorm = " << res_norm << "; errNorm = " << err_norm << std::endl;


	delete M;
}


template<class T, class Device>
Convergence<T, Device>* alloc_convergence(double tol, size_t max_restarts, int restart_length, double restart_tolerance, bool repeat_iter_restart, bool orth_loss_restart) {
	if (restart_tolerance == 0) {
		return new Convergence<T, Device>(tol, restart_length, max_restarts);
	} else if (repeat_iter_restart) {
		return new RepeatIteration_Convergence<T, Device>(tol, restart_tolerance, restart_length, max_restarts);
	} else if (orth_loss_restart) {
		return new LostOrthogonality_Convergence<T, Device>(tol, restart_tolerance, restart_length, max_restarts);
	} else {
		return new RelPrecRes_Convergence<T, Device>(tol, restart_tolerance, restart_length, max_restarts);
	}
}



template<class Device>
void run_tests(SparseMatrix<double, MKL> A_host,
			   Vect<double, MKL> x_host,
			   Vect<double, MKL> b_host,
	           int restart_length,
			   double restart_tolerance,
			   bool orth_loss_restart,
			   bool repeat_iter_restart,
			   double tol,
			   size_t max_restarts,
			   test_mode_t mode,
			   orth_t orth,
			   prec_t prec_type,
			   int jacobi_steps) {

	SparseMatrix<double, Device> A_double (A_host);
	const int n = A_double.nrows();

	Vect<double, Device> x (n);
	Kokkos::deep_copy(x.view(), x_host.view());
	Vect<double, Device> b (n);
	Kokkos::deep_copy(b.view(), b_host.view());

	std::cout << "||x|| = " << nrm2(x) << std::endl;
	std::cout << "||b|| = " << nrm2(b) << std::endl;
	std::cout << "||A|| = " << nrm2(A_host.vals_vect()) << std::endl;


	switch(mode) {
	case mixed_mode: {
		auto convergence = alloc_convergence<float, Device>(tol, max_restarts, restart_length, restart_tolerance, repeat_iter_restart, orth_loss_restart);
		switch(orth) {
		case cgs:
			DoMixedPrecisionProblem<Orthogonalization::GS<float, Orthogonalization::CGS_Kernel<float, Device>, Device>, Device>
					(*convergence, A_double, b, x, prec_type, jacobi_steps);
			break;
		case mgs:
			DoMixedPrecisionProblem<Orthogonalization::GS<float, Orthogonalization::MGS_Kernel<float, Device>, Device>, Device>
					(*convergence, A_double, b, x, prec_type, jacobi_steps);
			break;
		case cgsr:
			DoMixedPrecisionProblem<Orthogonalization::GS<float, Orthogonalization::CGSR_Kernel<float, Device, 2>, Device>, Device>
					(*convergence, A_double, b, x, prec_type, jacobi_steps);
			break;
		}
		delete convergence;
		break;
	}
	case baseline_mode: {
		auto convergence = alloc_convergence<double, Device>(tol, max_restarts, restart_length, restart_tolerance, repeat_iter_restart, orth_loss_restart);
		switch(orth) {
		case cgs:
			DoBaselineProblem<Orthogonalization::GS<double, Orthogonalization::CGS_Kernel<double, Device>, Device>, Device, double, double>
					(*convergence, A_double, b, x, prec_type, jacobi_steps);
			break;
		case mgs:
			DoBaselineProblem<Orthogonalization::GS<double, Orthogonalization::MGS_Kernel<double, Device>, Device>, Device, double, double>
					(*convergence, A_double, b, x, prec_type, jacobi_steps);
			break;
		case cgsr:
			DoBaselineProblem<Orthogonalization::GS<double, Orthogonalization::CGSR_Kernel<double, Device, 2>, Device>, Device, double, double>
					(*convergence, A_double, b, x, prec_type, jacobi_steps);
			break;
		}
		delete convergence;
		break;
	}
	case single_prec_mode: {
		auto convergence = alloc_convergence<double, Device>(tol, max_restarts, restart_length, restart_tolerance, repeat_iter_restart, orth_loss_restart);
		switch(orth) {
		case cgs:
			DoBaselineProblem<Orthogonalization::GS<double, Orthogonalization::CGS_Kernel<double, Device>, Device>, Device, double, float>
					(*convergence, A_double, b, x, prec_type, jacobi_steps);
			break;
		case mgs:
			DoBaselineProblem<Orthogonalization::GS<double, Orthogonalization::MGS_Kernel<double, Device>, Device>, Device, double, float>
					(*convergence, A_double, b, x, prec_type, jacobi_steps);
			break;
		case cgsr:
			DoBaselineProblem<Orthogonalization::GS<double, Orthogonalization::CGSR_Kernel<double, Device, 2>, Device>, Device, double, float>
					(*convergence, A_double, b, x, prec_type, jacobi_steps);
			break;
		}
		delete convergence;
		break;
	}
	case single_mode: {
		auto convergence = alloc_convergence<float, Device>(tol, max_restarts, restart_length, restart_tolerance, repeat_iter_restart, orth_loss_restart);
		switch(orth) {
		case cgs:
			DoBaselineProblem<Orthogonalization::GS<float, Orthogonalization::CGS_Kernel<float, Device>, Device>, Device, float, float>
					(*convergence, A_double, b, x, prec_type, jacobi_steps);
			break;
		case mgs:
			DoBaselineProblem<Orthogonalization::GS<float, Orthogonalization::MGS_Kernel<float, Device>, Device>, Device, float, float>
					(*convergence, A_double, b, x, prec_type, jacobi_steps);
			break;
		case cgsr:
			DoBaselineProblem<Orthogonalization::GS<float, Orthogonalization::CGSR_Kernel<float, Device, 2>, Device>, Device, float, float>
					(*convergence, A_double, b, x, prec_type, jacobi_steps);
			break;
		}
		delete convergence;
		break;
	}
	}
}


int main(int argc, char* argv[]) {

	char* APath = nullptr;
	char* bPath = nullptr;
	int restart_length = 0;
	double restart_tolerance = 0;
	bool orth_loss_restart = false;
	bool repeat_iter_restart = false;
	double tol = 1e-6;
	size_t max_restarts = 1000000;
	int rand_seed = 42;
	orth_t orth = mgs;
	test_mode_t mode = mixed_mode;
	prec_t prec_type = ilu;
	int jacobi_steps = 1;

	bool use_gpu = false;

	for (int i = 1; i < argc; i++) {
		if (strcmp("--Apath", argv[i]) == 0) {
			APath = argv[++i];
		} else if (strcmp("--bpath", argv[i]) == 0) {
			bPath = argv[++i];
		} else if (strcmp("--rlen", argv[i])  == 0) {
			restart_length = std::stoi(argv[++i]);
		} else if (strcmp("--rtol", argv[i]) == 0) {
			restart_tolerance = std::stod(argv[++i]);
		} else if (strcmp("--repeat-iter", argv[i]) == 0) {
			repeat_iter_restart = true;
		} else if (strcmp("--orthloss", argv[i]) == 0) {
			orth_loss_restart = true;
		} else if (strcmp("--tol", argv[i]) == 0) {
			tol = std::stod(argv[++i]);
		} else if (strcmp("--max-restarts", argv[i]) == 0) {
			max_restarts = std::stol(argv[++i]);
		} else if (strcmp("--rand", argv[i]) == 0) {
			rand_seed = std::stoi(argv[++i]);
		} else if (strcmp("--mode", argv[i]) == 0) {
			++i;
			if (strcmp("mixed", argv[i]) == 0) {
				mode = mixed_mode;
			} else if (strcmp("baseline", argv[i]) == 0) {
				mode = baseline_mode;
			} else if (strcmp("single-prec", argv[i]) == 0) {
				mode = single_prec_mode;
			} else if (strcmp("single", argv[i]) == 0) {
				mode = single_mode;
			} else {
				std::cout << "Unknown test mode" << std::endl;
				return 1;
			}
		} else if (strcmp("--orth", argv[i]) == 0) {
			i++;
			if (strcmp("cgs", argv[i]) == 0) {
				orth = cgs;
			} else if (strcmp("mgs", argv[i]) == 0) {
				orth = mgs;
			} else if (strcmp("cgsr", argv[i]) == 0) {
				orth = cgsr;
			} else {
				std::cout << "Unknown Orthogonalization" << std::endl;
				return 1;
			}
		} else if (strcmp("--prec", argv[i]) == 0) {
			++i;
			if (strcmp("ilu", argv[i]) == 0) {
				prec_type = ilu;
			} else if (strcmp("identity", argv[i]) == 0) {
				prec_type = identity;
			} else if (strcmp("jacobi", argv[i]) == 0) {
				prec_type = jacobi;
			} else if (strcmp("ilu_jacobi", argv[i]) == 0) {
				prec_type = ilu_jacobi;
			} else {
				std::cout << "Unknown Preconditioner" << std::endl;
				return 1;
			}
		} else if (strcmp("--jacobi-steps", argv[i]) == 0) {
			jacobi_steps = std::stoi(argv[++i]);
		} else if (strcmp("--gpu", argv[i]) == 0) {
			use_gpu = true;
		} else {
			std::cout << "Unknown flag" << argv[i] << std::endl;
			return 1;
		}
	}

	if (repeat_iter_restart && orth_loss_restart) {
		std::cout << "Repeated Iteration Restart cannot be used with OrthLoss restart" << std::endl;
		return 1;
	}

	if (APath == nullptr) {
		std::cout << "No value suplied for A" << std::endl;
		return 1;
	}

	Kokkos::ScopeGuard kokkos(argc, argv);

	const SparseMatrix<double, MKL> A_host = LoadMatrix<double>(APath);
	const int n = A_host.nrows();
	Vect<double, MKL> x_host;
	Vect<double, MKL> b_host;

	if (bPath == nullptr) {
		x_host = rand_vect(n, rand_seed);
		b_host = Vect<double, MKL>(n);
		spmv(1.0, A_host, x_host, 0.0, b_host);
	} else {
		x_host = Vect<double, MKL>(n);
		fill(0.0, x_host);
		b_host = LoadVector<double>(bPath);
	}

	if (use_gpu) {
		run_tests<Cuda>(A_host,
					    x_host,
			            b_host,
			            restart_length,
					    restart_tolerance,
					    orth_loss_restart,
					    repeat_iter_restart,
					    tol,
					    max_restarts,
					    mode,
					    orth,
					    prec_type,
						jacobi_steps);
	} else {
		run_tests<MKL>(A_host,
					   x_host,
			           b_host,
			           restart_length,
					   restart_tolerance,
					   orth_loss_restart,
					   repeat_iter_restart,
					   tol,
					   max_restarts,
					   mode,
					   orth,
					   prec_type,
					   jacobi_steps);
	}


	return 0;
}
