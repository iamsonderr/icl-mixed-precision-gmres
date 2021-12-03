#ifndef ORTHOGONALIZATION_HPP_
#define ORTHOGONALIZATION_HPP_

#include "kernels.hpp"

namespace Orthogonalization {

template<class Type, class Device>
class Orth {
public:
	virtual MultiVect<Type, Device> basis() = 0;

	virtual ~Orth() {
	}
};

template<class Type, class GS_Kernel, class Device>
class GS: public Orth<Type, Device> {
private:

	GS_Kernel kernel;

public:

	MultiVect<Type, Device> v;

	GS(size_t n, size_t max_restart_length)
		: v(n, max_restart_length+1),
		  kernel(n, max_restart_length) {
	}

	MultiVect<Type, Device> basis() {
		return v;
	}

	Type first_vector(const Vect<Type, Device> w) {
		const Type beta = nrm2(w);
		Vect<Type, Device> v_col (v, Kokkos::ALL, 0);
		if (beta != 0) {
			scal(1/beta, w, v_col);
		} else {
			fill(0.0, v_col);
		}
		return beta;
	}

	Vect<Type, Device> previous_krylov_vector(size_t k) {
		return Vect<Type, Device>(v, Kokkos::ALL, k);
	}

	void add_vector(const size_t k, Vect<Type, Device> w, MultiVect<Type, Device> h) {

		kernel.orthogonalize(v, k, w, h);

		nrm2(w, h(k+1, k));
		auto h_final = h(k+1, k).access();

		Vect<Type, Device> v_col (v, Kokkos::ALL, k+1);
		scal(1/h_final, w, v_col);
	}

	void update_x(const size_t k, const Vect<Type, Device> y, Vect<Type, Device> x) const {
		MultiVect<Type, Device> v_cols (v, Kokkos::ALL, Kokkos::pair<int, int>(0, k));
		gemv(1.0, v_cols, y, 1.0, x);
	}

	template<class High>
	void update_x(const size_t k, const Vect<Type, Device> y, Vect<High, Device> x, Vect<Type, Device> x_inc_temp, Vect<High, Device> x_temp) const {
		MultiVect<Type, Device> v_cols (v, Kokkos::ALL, Kokkos::pair<int, int>(0, k));
		gemv(1.0, v_cols, y, 0.0, x_inc_temp);
		copy(x_inc_temp, x_temp);
		axpy(1.0, x_temp, x);
	}
};

template<class Type, class Device>
class CGS_Kernel {
public:
	CGS_Kernel(size_t, size_t) {
	}

	void orthogonalize(MultiVect<Type, Device> v, const size_t k, Vect<Type, Device> w, MultiVect<Type, Device> h) const {
		MultiVect<Type, Device> v_prevCols (v, Kokkos::ALL, Kokkos::pair<int, int>(0, k+1));
		Vect<Type, Device> h_col (h, Kokkos::pair<size_t, size_t>(0, k+1), k);

		gemv(1.0, v_prevCols.transpose_matrix(), w, 0.0, h_col);
		gemv(-1.0, v_prevCols, h_col, 1.0, w);
	}
};

template<class Type, class Device>
class MGS_Kernel {
public:

	MGS_Kernel(size_t, size_t) {
	}

	void orthogonalize(MultiVect<Type, Device> v, size_t k, Vect<Type, Device> w, MultiVect<Type, Device> h) const {
		for (size_t j = 0; j < k+1; ++j) {
			Vect<Type, Device> v_col (v, Kokkos::ALL, j);

			dot(w, v_col, h(j, k));

			naxpy(h(j, k), v_col, w);
		}
	}
};

template<class Type, class Device, size_t orth_steps>
class CGSR_Kernel {
private:

	Vect<Type, Device> weights;

public:
	CGSR_Kernel(size_t, size_t max_restart_length)
		: weights(max_restart_length) {
	}

	void orthogonalize(MultiVect<Type, Device> v, const size_t k, Vect<Type, Device> w, MultiVect<Type, Device> h) const {
		MultiVect<Type, Device> v_prevCols (v, Kokkos::ALL, Kokkos::pair<int, int>(0, k+1));
		Vect<Type, Device> weights_view (weights, Kokkos::pair<size_t, size_t>(0, k+1));
		Vect<Type, Device> h_col (h, Kokkos::pair<size_t, size_t>(0, k+1), k);

		// first iteration can use h_col directly
		gemv(1.0, v_prevCols.transpose_matrix(), w, 0.0, h_col);
		gemv(-1.0, v_prevCols, h_col, 1.0, w);

		for (size_t i = 1; i < orth_steps; i++){
			gemv(1.0, v_prevCols.transpose_matrix(), w, 0.0, weights_view);
			gemv(-1.0, v_prevCols, weights_view, 1.0, w);

			axpy(1.0, weights_view, h_col);
		}
	}
};

} // namespace Orthogonalization


#endif /* ORTHOGONALIZATION_HPP_ */
