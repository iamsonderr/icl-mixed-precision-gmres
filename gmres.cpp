
#include <Kokkos_Core.hpp>

#include "types_mkl.hpp"
#include "types_cuda.hpp"
#include "kernels.hpp"

#include "gmres.hpp"

#include <iostream>

template<class Type, class Device, class RHSType>
void typesafe_apply(LinearOperator<Type, Device>* op, Vect<RHSType, Device> rhs, Vect<Type, Device> temp) {
    copy(rhs, temp);
    op->apply(temp);
    copy(temp, rhs);
}

template<class Type, class Device>
void typesafe_apply(LinearOperator<Type, Device>* op, Vect<Type, Device> rhs, Vect<Type, Device> temp) {
    op->apply(rhs);
}

template<class Orth, class Device, class Type, class PrecType>
void gmres_baseline(Convergence<Type, Device>& convergence,
					SparseMatrix<Type, Device> A,
					LinearOperator<PrecType, Device>* M,
					Vect<Type, Device> b,
					Vect<Type, Device> x) {

	const size_t n = x.n();

	const size_t max_restart_length = convergence.max_restart_length();

	Orth orth (n, max_restart_length);

	// vectors for solving the least squares problem
	Vect<Type, Device> cos(max_restart_length+1);
	Vect<Type, Device> sin(max_restart_length+1);
	Vect<Type, Device> s(max_restart_length+1);

	// vectors and multi-vectors for building the orthogonal basis
	Vect<Type, Device> w (n);
	MultiVect<Type, Device> v (n, max_restart_length+1);
	MultiVect<Type, Device> h (max_restart_length+1, max_restart_length);

	Vect<PrecType, Device> w_temp (std::is_same<PrecType, Type>::value ? 0 : n);

	convergence.setup(orth);

	const Type b_norm = nrm2(b);

	copy(b, w);
	typesafe_apply(M, w, w_temp);
	const Type Minvb_norm = nrm2(w);

	const Type A_norm = nrm2(A.vals_vect());

	for (size_t i = 0; true; ++i) {

		// Mr = -Ax + b
		copy(b, w);
		spmv(-1.0, A, x, 1.0, w);
		const Type r_norm = nrm2(w);
		typesafe_apply(M, w, w_temp);

		auto beta = nrm2(w);

		const Type x_norm = nrm2(x);
		// Check for convergence or restart
		switch(convergence.check_initial(r_norm, b_norm+A_norm*x_norm, beta, Minvb_norm)) {
		case iteration_converged:
			std::cout << "Found solution with rel prec res norm = " << Type(beta/Minvb_norm) << " when k = 0 and i = " << i << std::endl;
			std::cout << "  total iterations = " << convergence.total_iterations() << std::endl;
			return;
		case iteration_aborted:
			std::cout << "Aborting after " << convergence.total_iterations() << " iterations" << std::endl;
			return;
		default:
			break;
		}

		orth.first_vector(w);

		auto s_view = s.view();
		Kokkos::parallel_for(Kokkos::RangePolicy<typename Device::execution_space>(0, s.n()),
				             KOKKOS_LAMBDA(const size_t& i) {
			if (i == 0) {
				s_view(i) = beta;
			} else {
				s_view(i) = 0;
			}
		});

		bool continue_inner_iter = true;
		size_t k;
		for (k = 0; continue_inner_iter; ++k) {
			{ //new scope for v_col
				Vect<Type, Device> v_col = orth.previous_krylov_vector(k);
				spmv(1.0, A, v_col, 0.0, w);
				typesafe_apply(M, w, w_temp);
			}

			orth.add_vector(k, w, h);

			auto range  = Kokkos::pair<size_t, size_t>(0, k);
			auto range1 = Kokkos::pair<size_t, size_t>(0, k+1);
			rot(h(range1, k), cos(range), sin(range));
			rotg(h(k, k), h(k+1, k), cos(k), sin(k));
			rot (s(k),    s(k+1),    cos(k), sin(k));

			// Check for convergence or restart
			typename Device::execution_space().fence();
			Type arnoldi_residual = std::fabs(s.access(k+1));
			switch(convergence.check(k+1, arnoldi_residual, Minvb_norm)) {
			case iteration_converged:
				solution_update(orth, x, k+1, h, s);
				std::cout << "Found solution with rel prec res norm = " << arnoldi_residual/Minvb_norm << " when k = " << k+1 << " and i = " << i << std::endl;
				std::cout << "  total iterations = " << convergence.total_iterations() << std::endl;
				return;
			case iteration_restart:
				continue_inner_iter = false;
				break;
			case iteration_next:
				break;
			case iteration_aborted:
				std::cout << "Aborting after " << convergence.total_iterations() << " iterations" << std::endl;
				return;
			}
		}
		solution_update(orth, x, k, h, s);
	}
}

template<class Orth, class Device>
void gmres_singleUpdate(Convergence<float, Device>& convergence,
						SparseMatrix<double, Device> A,
						SparseMatrix<float, Device> A_single,
						LinearOperator<float, Device>* M,
						Vect<double, Device> b,
						Vect<double, Device> x) {

	const size_t n = x.n();

	const size_t max_restart_length = convergence.max_restart_length();

	Orth orth (n, max_restart_length);

	// vectors for solving the least squares problem
	Vect<float, Device> cos(max_restart_length+1); // Also used as a temp in Update
	Vect<float, Device> sin(max_restart_length+1);
	Vect<float, Device> s( max_restart_length+1);

	// vectors and multi-vectors for building the orthogonal basis
	Vect<float, Device> w (n);
	MultiVect<float, Device> v (n, max_restart_length+1);
	MultiVect<float, Device> h (max_restart_length+1, max_restart_length);
	Vect<double, Device> r_accum (n); // Also used as a temp in Update

	convergence.setup(orth);

	const double b_norm = nrm2(b);
	copy(b, w);
	M->apply(w);
	const double Minvb_norm = nrm2(w);

	// Frobenius norm
	const double A_norm = nrm2(A_single.vals_vect());

	for (size_t i = 0; true; ++i) {

		// Mr = -Ax + b
		copy(b, r_accum);
		spmv(-1.0, A, x, 1.0, r_accum);
		copy(r_accum, w);
		const double r_norm = nrm2(w);
		M->apply(w);

		auto beta = nrm2(w);

		const double x_norm = nrm2(x);

		// Check for convergence or restart
		switch(convergence.check_initial(r_norm, b_norm+A_norm*x_norm, beta, Minvb_norm)) {
		case iteration_converged:
			std::cout << "Found solution with rel prec res norm = " << double(beta/Minvb_norm) << " when k = 0 and i = " << i << std::endl;
			std::cout << "  total iterations = " << convergence.total_iterations() << std::endl;
			return;
		case iteration_aborted:
			std::cout << "Aborting after " << convergence.total_iterations() << " iterations" << std::endl;
			return;
		default:
			break;
		}

		orth.first_vector(w);

		auto s_view = s.view();
		Kokkos::parallel_for(Kokkos::RangePolicy<typename Device::execution_space>(0, s.n()),
				             KOKKOS_LAMBDA(const size_t& i) {
			if (i == 0) {
				s_view(i) = beta;
			} else {
				s_view(i) = 0;
			}
		});

		bool continue_inner_iter = true;
		size_t k;
		for (k = 0; continue_inner_iter; ++k) {
			{ //new scope for v_col
				Vect<float, Device> v_col = orth.previous_krylov_vector(k);
				spmv(1.0, A_single, v_col, 0.0, w);
				M->apply(w);
			}

			orth.add_vector(k, w, h);

			auto range = Kokkos::pair<size_t, size_t>(0, k);
			rot(h(range, k), cos(range), sin(range));
			rotg(h(k, k), h(k+1, k), cos(k), sin(k));
			rot (s(k),    s(k+1),    cos(k), sin(k));

			// Check for convergence or restart
			typename Device::execution_space().fence();
			double arnoldi_residual = std::fabs(s.access(k+1));
			switch(convergence.check(k+1, arnoldi_residual, Minvb_norm)) {
			case iteration_converged:
				solution_update(orth, x, k+1, h, s, w, r_accum);
				std::cout << "Found solution with rel prec res norm = " << arnoldi_residual/Minvb_norm << " when k = " << k+1 << " and i = " << i << std::endl;
				std::cout << "  total iterations = " << convergence.total_iterations() << std::endl;
				return;
			case iteration_restart:
				continue_inner_iter = false;
				break;
			case iteration_next:
				break;
			case iteration_aborted:
				std::cout << "Aborting after " << convergence.total_iterations() << " iterations" << std::endl;
				return;
			}
		}
		solution_update(orth, x, k, h, s, w, r_accum);
	}
}

template<class Orth, class Device>
double compute_backwards_error(
					const SparseMatrix<double, Device>& A,
					const Vect<double, Device>& x,
					const Vect<double, Device>& b,
					const double a_norm,
					const double b_norm,
					Orth& orth,
					size_t k,
					const MultiVect<double, Device>& h, const Vect<double, Device>& s,
					Vect<double, Device>& temp_r) {
	size_t n = temp_r.n();
	Vect<double, Device> x_temp (n);
	copy(x, x_temp);

	Vect<double, Device> s_temp (s.n());
	copy(s, s_temp);

	copy(b, temp_r);

	solution_update(orth, x_temp, k, h, s_temp);
	spmv(-1.0, A, x_temp, 1.0, temp_r);

	const double x_norm = nrm2(x_temp);
	const double r_norm = nrm2(temp_r);

	return r_norm / (b_norm + a_norm*x_norm);
}

template<class Orth, class Device>
void solution_update(Orth &orth,
			Vect<double, Device> &x,
			const size_t k,
			const MultiVect<float, Device> h,
			const Vect<float, Device> s,
			Vect<float, Device> x_inc_temp,
			Vect<double, Device> x_temp) {

	Vect<float, Device> y(s, std::make_pair(size_t(0), k));
	MultiVect<float, Device> h_temp(h, std::make_pair(size_t(0), k), std::make_pair(size_t(0), k));

	trsv("Upper", h_temp, y);
	orth.update_x(k, y, x, x_inc_temp, x_temp);
}
template<class Orth, class Device, class Type>
void solution_update(Orth &orth,
					 Vect<Type, Device> &x,
					 const size_t k,
					 const MultiVect<Type, Device> h,
					 const Vect<Type, Device> s) {
	Vect<Type, Device> y(s, std::make_pair(size_t(0), k));
	MultiVect<Type, Device> h_temp(h, std::make_pair(size_t(0), k), std::make_pair(size_t(0), k));

	trsv("Upper", h_temp, y);

	orth.update_x(k, y, x);
}


#define UNPACK( ... ) __VA_ARGS__

#define CREATE_BASELINE_CONFIGS(DEVICE, ORTH_KERNEL, TYPE, PREC_TYPE) \
template void gmres_baseline<Orthogonalization::GS<TYPE, UNPACK ORTH_KERNEL, DEVICE>, DEVICE, TYPE, PREC_TYPE> \
					(Convergence<TYPE, DEVICE>& convergence, \
					 SparseMatrix<TYPE, DEVICE> A, \
					 LinearOperator<PREC_TYPE, DEVICE>* M, \
					 Vect<TYPE, DEVICE> b, \
					 Vect<TYPE, DEVICE> x);

 #define CREATE_MP_CONFIGS(DEVICE, ORTH_KERNEL) \
 template void gmres_singleUpdate<Orthogonalization::GS<float, UNPACK ORTH_KERNEL, DEVICE>, DEVICE> \
 					(Convergence<float, DEVICE>& convergence, \
 					 SparseMatrix<double, DEVICE> A, \
					 SparseMatrix<float, DEVICE> A_single, \
 					 LinearOperator<float, DEVICE>* M, \
 					 Vect<double, DEVICE> b, \
 					 Vect<double, DEVICE> x);

#define CREATE_SOLUTION_UPDATES(DEVICE, ORTH_KERNEL_D, ORTH_KERNEL_S) \
template void solution_update<Orthogonalization::GS<double, UNPACK ORTH_KERNEL_D, DEVICE>, DEVICE, double> \
					(Orthogonalization::GS<double, UNPACK ORTH_KERNEL_D, DEVICE>& orth, \
					 Vect<double, DEVICE> &x, \
					 size_t k, \
					 MultiVect<double, DEVICE> h, \
					 Vect<double, DEVICE> s); \
template void solution_update<Orthogonalization::GS<float, UNPACK ORTH_KERNEL_S, DEVICE>, DEVICE, float> \
					(Orthogonalization::GS<float, UNPACK ORTH_KERNEL_S, DEVICE>& orth, \
					 Vect<float, DEVICE> &x, \
					 size_t k, \
					 MultiVect<float, DEVICE> h, \
					 Vect<float, DEVICE> s); \
template void solution_update<Orthogonalization::GS<float, UNPACK ORTH_KERNEL_S, DEVICE>, DEVICE> \
					(Orthogonalization::GS<float, UNPACK ORTH_KERNEL_S, DEVICE>& orth, \
					 Vect<double, DEVICE> &x, \
					 size_t k, \
					 MultiVect<float, DEVICE> h, \
					 Vect<float, DEVICE> s, \
					 Vect<float, DEVICE> x_inc_temp, \
					 Vect<double, DEVICE> x_temp);

#define CREATE_TYPE_CONFIGS(DEVICE, ORTH_KERNEL_D, ORTH_KERNEL_S) \
CREATE_BASELINE_CONFIGS(DEVICE, (Orthogonalization::UNPACK ORTH_KERNEL_D), double, double) \
CREATE_BASELINE_CONFIGS(DEVICE, (Orthogonalization::UNPACK ORTH_KERNEL_D), double, float) \
CREATE_BASELINE_CONFIGS(DEVICE, (Orthogonalization::UNPACK ORTH_KERNEL_S), float, float) \
CREATE_MP_CONFIGS(DEVICE,       (Orthogonalization::UNPACK ORTH_KERNEL_S)) \
CREATE_SOLUTION_UPDATES(DEVICE, (Orthogonalization::UNPACK ORTH_KERNEL_D), (Orthogonalization::UNPACK ORTH_KERNEL_S))

#define CREATE_TEST_CONFIGS(DEVICE) \
CREATE_TYPE_CONFIGS(DEVICE, (CGS_Kernel<double, DEVICE>),     (CGS_Kernel<float, DEVICE>)) \
CREATE_TYPE_CONFIGS(DEVICE, (MGS_Kernel<double, DEVICE>),     (MGS_Kernel<float, DEVICE>)) \
CREATE_TYPE_CONFIGS(DEVICE, (CGSR_Kernel<double, DEVICE, 2>), (CGSR_Kernel<float, DEVICE, 2>)) \

CREATE_TEST_CONFIGS(MKL)
CREATE_TEST_CONFIGS(Cuda)
