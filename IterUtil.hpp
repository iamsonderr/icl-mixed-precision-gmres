
#ifndef ITERUTIL_HPP_
#define ITERUTIL_HPP_

#include <cstddef>

#include "kernels.hpp"
#include "Orthogonalization.hpp"

enum iteration_action {
	iteration_next,
	iteration_converged,
	iteration_restart,
	iteration_aborted
};

template<class T, class Device>
class Convergence {
public:
	const double tol;
	const size_t restart_length;
	const size_t max_restarts;

	size_t total_iters = 0;
	size_t total_restarts = 0;

	Convergence(double tol, size_t restart_length, size_t max_restarts)
		: tol(tol), restart_length(restart_length), max_restarts(max_restarts) {
	}

	/**
	 * Initialize the Convergence object with the data structures from GMRES
	 */
	virtual void setup(Orthogonalization::Orth<T, Device>&) {
		total_iters = 0;
	}
	/**
	 * Check the first vector in an outer iteration for convergence
	 * First and second arguments are the residual norm, and b norm
	 * Third and fourth arguments are the preconditioned residual norm, and preconditioned b norm respectively
	 */
	virtual iteration_action check_initial(double residualNorm, double normalization, double, double) {
		total_restarts++;
		if (total_restarts > max_restarts) {
			return iteration_aborted;
		} else if (residualNorm/normalization > tol) {
			return iteration_next;
		} else {
			return iteration_converged;
		}
	}
	/**
	 * Check the next vector for convergence
	 * First argument is the index of the inner iteration
	 * Second and third arguments are the preconditioned residual norm, and preconditioned b norm respectively
	 */
	virtual iteration_action check(size_t k, double, double) {
		total_iters++;

		if (restart_length <= k) {
			return iteration_restart;
		}

		return iteration_next;
	}
	/**
	 * Get the maximimum number of iterations in an inner iteration
	 */
	virtual size_t max_restart_length() const {
		return restart_length;
	}
	/**
	 * Get the total iterations run so far
	 */
	virtual size_t total_iterations() const {
		return total_iters;
	}

	virtual ~Convergence() {
	}
};


template<class T, class Device>
class RepeatIteration_Convergence: public Convergence<T, Device> {
private:
	const double restart_improvement;
	double restart_tol;

	size_t second_restart_length = 0;

	bool first_iteration = true;
public:
	RepeatIteration_Convergence(double tol, double restart_improvement, size_t restart_length, size_t max_restarts)
		: Convergence<T, Device>(tol, restart_length, max_restarts),
		  restart_improvement(restart_improvement), restart_tol(restart_improvement) {
	}

	virtual iteration_action check_initial(double residualNorm, double normalization, double precResidualNorm, double precbnorm) {
		if (first_iteration) {
			restart_tol = precResidualNorm/precbnorm * restart_improvement;
		}
		return Convergence<T, Device>::check_initial(residualNorm, normalization, precResidualNorm, precbnorm);
	}

	virtual iteration_action check(size_t k, double residualNorm, double bnorm) {
		if (first_iteration) {
			const iteration_action super_action = Convergence<T, Device>::check(k, residualNorm, bnorm);
			if (super_action != iteration_next) {
				first_iteration = false;
				second_restart_length = k;
				return super_action;
			}

			if (residualNorm/bnorm <= restart_tol) {
				first_iteration = false;
				second_restart_length = k;
				return iteration_restart;
			}
			return iteration_next;
		} else {
			const iteration_action super_action = Convergence<T, Device>::check(k, residualNorm, bnorm);
			if (super_action != iteration_next) {
				return super_action;
			}

			if (second_restart_length <= k) {
				return iteration_restart;
			}

			return iteration_next;
		}
	}

	virtual ~RepeatIteration_Convergence() {
	}
};

template<class T, class Device>
class RelPrecRes_Convergence: public Convergence<T, Device> {
private:
	const double restart_improvement;
	double restart_tol;
public:
	RelPrecRes_Convergence(double tol, double restart_improvement, size_t restart_length, size_t max_restarts)
		: Convergence<T, Device>(tol, restart_length, max_restarts),
		  restart_improvement(restart_improvement), restart_tol(restart_improvement) {
	}

	virtual iteration_action check_initial(double residualNorm, double normalization, double precResidualNorm, double precbnorm) {
		restart_tol = precResidualNorm/precbnorm * restart_improvement;
		return Convergence<T, Device>::check_initial(residualNorm, normalization, precResidualNorm, precbnorm);
	}

	virtual iteration_action check(size_t k, double residualNorm, double bnorm) {
		const iteration_action super_action = Convergence<T, Device>::check(k, residualNorm, bnorm);
		if (super_action != iteration_next) {
			return super_action;
		}

		if (residualNorm/bnorm <= restart_tol) {
			return iteration_restart;
		}
		return iteration_next;
	}

	virtual ~RelPrecRes_Convergence() {
	}
};


template<class T, class Device>
class LostOrthogonality_Convergence: public Convergence<T, Device> {
private:
	const double restart_tol_squared;

	double current_loss_squared = 0;
	MultiVect<T, Device> S;
	Vect<T, Device> u;
	MultiVect<T, Device> v;
public:
	LostOrthogonality_Convergence(double tol, double restart_tol, size_t restart_length, size_t max_restarts)
		: Convergence<T, Device>(tol, restart_length, max_restarts),
		  restart_tol_squared(restart_tol*restart_tol),
		  S(restart_length+1, restart_length+1),
		  u(restart_length+1) {
	}

	virtual void setup(Orthogonalization::Orth<T, Device>& orth) {
		this->v = orth.basis();
		fill(0.0, S);
		Convergence<T, Device>::setup(orth);
	}

	virtual iteration_action check_initial(double residualNorm, double normalization, double precResidualNorm, double precbnorm) {
		current_loss_squared = 0;
		return Convergence<T, Device>::check_initial(residualNorm, normalization, precResidualNorm, precbnorm);
	}

	virtual iteration_action check(size_t k, double precResidualNorm, double precbnorm) {
		const iteration_action super_action = Convergence<T, Device>::check(k, precResidualNorm, precbnorm);
		if (super_action != iteration_next) {
			return super_action;
		}

		Vect<T, Device> u_short = Vect<T, Device>(u, std::pair<int, int>(0, k+1));
		Vect<T, Device> v_col = Vect<T, Device>(v, Kokkos::ALL, k+1);
		MultiVect<T, Device> v_prevCols = MultiVect<T, Device>(v, Kokkos::ALL, std::pair<int, int>(0, k+1));
		gemv(1.0, v_prevCols.transpose_matrix(), v_col, 0.0, u_short);

		Vect<T, Device> s_col = Vect<T, Device>(S, std::pair<int, int>(0, k+1), k+1);
		MultiVect<T, Device> S_prevCols = MultiVect<T, Device>(S, std::pair<int, int>(0, k+1), std::pair<int, int>(0, k+1));
		copy(u_short, s_col);
		gemv(-1.0, S_prevCols, u_short, 1.0, s_col);

		current_loss_squared += dot(s_col, s_col);

		if (current_loss_squared >= restart_tol_squared) {
			return iteration_restart;
		}

		return iteration_next;
	}

	virtual ~LostOrthogonality_Convergence() {
	}
};

#endif /* ITERUTIL_HPP_ */
