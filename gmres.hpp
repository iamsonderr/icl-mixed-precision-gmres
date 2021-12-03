
#ifndef GMRES_HPP_
#define GMRES_HPP_

#include <utility>

#include "IterUtil.hpp"
#include "types.hpp"

#include <iostream>

/**
 * The baseline, double precision implementation for GMRES
 */
template<class Orth, class Device, class Type, class PrecType>
void gmres_baseline(Convergence<Type, Device>& convergence,
					SparseMatrix<Type, Device> A,
					LinearOperator<PrecType, Device>* M,
					Vect<Type, Device> b,
					Vect<Type, Device> x);

/**
 * A mixed precision implementation for GMRES.  The residual is computed in
 * double precision, while the update is computed in single precision
 */
template<class Orth, class Device>
void gmres_singleUpdate(Convergence<float, Device>& convergence,
						SparseMatrix<double, Device> A,
						SparseMatrix<float, Device> A_single,
						LinearOperator<float, Device>* M,
						Vect<double, Device> b,
						Vect<double, Device> x);

template<class Orth, class Device>
void gmres_measure_iterations(Convergence<double, Device>& convergence,
							  SparseMatrix<double, Device> A,
							  LinearOperator<double, Device>* M,
							  Vect<double, Device> b,
							  Vect<double, Device> x);

/**
 * Applies the update to x
 */
template<class Orth, class Device>
void solution_update(Orth &orth,
			Vect<double, Device> &x,
			const size_t k,
			const MultiVect<float, Device> h,
			const Vect<float, Device> s,
			Vect<float, Device> x_inc_temp,
			Vect<double, Device> x_temp);
template<class Orth, class Device, class Type>
void solution_update(Orth &orth,
					 Vect<Type, Device> &x,
					 const size_t k,
					 const MultiVect<Type, Device> h,
					 const Vect<Type, Device> s);

#endif /* GMRES_HPP_ */
