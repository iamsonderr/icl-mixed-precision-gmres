
#ifndef KERNELS_HPP
#define KERNELS_HPP

#include<cassert>

#include "types.hpp"

// BLAS-1

template<class Type1, class Type2, class Device>
void copy(Vect<Type1, Device> x, Vect<Type2, Device> y) {
    assert(x.n() == y.n());
    auto x_view = x.view();
    auto y_view = y.view();
    Kokkos::parallel_for(Kokkos::RangePolicy<typename Device::execution_space>(0, x.n()),
                         KOKKOS_LAMBDA(const size_t& i) {
        y_view(i) = x_view(i);
    });
}

template<class Type1, class Type2, class Device>
void copy(Scalar<Type1, Device> x, Scalar<Type2, Device> y) {
    auto x_view = x.view();
    auto y_view = y.view();
    Kokkos::parallel_for(Kokkos::RangePolicy<typename Device::execution_space>(0, 1),
                         KOKKOS_LAMBDA(const size_t&) {
        y_view() = x_view();
    });
}


template<class Type, class Device>
Type dot(Vect<Type, Device> x, Vect<Type, Device> y);

template<class Type, class Device>
void dot(Vect<Type, Device> x, Vect<Type, Device> y, Scalar<Type, Device> result);


template<class Type, class Device>
Type nrm2(Vect<Type, Device> x);

template<class Type, class Device>
void nrm2(Vect<Type, Device> x, Scalar<Type, Device> result);


template<class Type, class Device>
void axpy(Type alpha, Vect<Type, Device> x, Vect<Type, Device> y);

template<class Type, class Device>
void axpy(Scalar<Type, Device> alpha, Vect<Type, Device> x, Vect<Type, Device> y);

template<class ScalarType, class Type, class Device>
void axpy(ScalarType alpha, Vect<Type, Device> x, Vect<Type, Device> y) {
    axpy(Type(alpha), x, y);
}

// -alpha*x + y
template<class Type, class Device>
void naxpy(Scalar<Type, Device> alpha, Vect<Type, Device> x, Vect<Type, Device> y);

template<class Type, class Device>
void scal(Type alpha, Vect<Type, Device> x);

template<class Type, class Device>
void scal(Type alpha, Vect<Type, Device> x, Vect<Type, Device> y);

template<class ScalarType, class Type, class Device>
void scal(ScalarType alpha, Vect<Type, Device> x, Vect<Type, Device> y) {
    scal(Type(alpha), x, y);
}

template<class Type, class Device>
void scal(Scalar<Type, Device> alpha, Vect<Type, Device> x, Vect<Type, Device> y);

template<class Type, class Device>
void scal(Type alpha, Scalar<Type, Device> x, Scalar<Type, Device> y);

template<class ScalarType, class Type, class Device>
void scal(ScalarType alpha, Scalar<Type, Device> x, Scalar<Type, Device> y) {
    scal(Type(alpha), x, y);
}

template<class Type, class Device>
void scal(Scalar<Type, Device> alpha, Scalar<Type, Device> x, Scalar<Type, Device> y);


template<class Type, class Device, class ScalarType>
void fill(ScalarType alpha, Scalar<Type, Device> x) {
  Kokkos::deep_copy(x.view(), Type(alpha));
}

template<class Type, class Device, class ScalarType>
void fill(ScalarType alpha, Vect<Type, Device> x) {
  Kokkos::deep_copy(x.view(), Type(alpha));
}

template<class Type, class Device, class ScalarType>
void fill(ScalarType alpha, MultiVect<Type, Device> x) {
    Kokkos::deep_copy(x.view(), Type(alpha));
}


template<class Type, class Device>
void rotg(Scalar<Type, Device> a, Scalar<Type, Device> b,
          Scalar<Type, Device> c, Scalar<Type, Device> s);


template<class Type, class Device>
void rot(Scalar<Type, Device> a, Scalar<Type, Device> b,
         Scalar<Type, Device> c, Scalar<Type, Device> s);

template<class Type, class Device>
void rot(Vect<Type, Device> a, Vect<Type, Device> c, Vect<Type, Device> s);

// BLAS-2

template<class ScalarType, class Type, class Device>
void gemv(ScalarType alpha, MultiVect<Type, Device> matrix, Vect<Type, Device> x,
          ScalarType beta, Vect<Type, Device> y) {
    gemv(Type(alpha), matrix, x, Type(beta), y);
}

template<class Type, class Device>
void gemv(Type alpha, MultiVect<Type, Device> matrix, Vect<Type, Device> x, Type beta, Vect<Type, Device> y);


template<class Type, class Device>
void trsv(const char* upper, MultiVect<Type, Device> matrix, Vect<Type, Device> x);

template<class Type, class Device>
void gdmv(Type alpha, Vect<Type, Device> diag, Vect<Type, Device> x,
          Type beta, Vect<Type, Device> y) {
    const auto n = diag.n();
    assert(n == x.n());
    assert(n == y.n());

    auto diag_data = diag.data();
    auto x_data = x.data();
    auto y_data = y.data();
    Kokkos::parallel_for(Kokkos::RangePolicy<typename Device::execution_space>(0, n),
                         KOKKOS_LAMBDA(const size_t& i) {
        y_data[i] = beta*y_data[i] + alpha*diag_data[i]*x_data[i];
    });
}

template<class ScalarType, class Type, class Device>
void gdmv(ScalarType alpha, Vect<Type, Device> diag, Vect<Type, Device> x,
          ScalarType beta, Vect<Type, Device> y) {
    gdmv(Type(alpha), diag, x, Type(beta), y);
}

// Sparse

template<class Type, class Device>
ILU<Type, Device> ilu0(SparseMatrix<double, Device> matrix);


template<class Type, class Device>
void spmv(Type alpha, SparseMatrix<Type, Device> matrix, Vect<Type, Device> x, Type beta, Vect<Type, Device> y);

template<class ScalarType, class Type, class Device>
void spmv(ScalarType alpha, SparseMatrix<Type, Device> matrix, Vect<Type, Device> x, ScalarType beta, Vect<Type, Device> y) {
    spmv(Type(alpha), matrix, x, Type(beta), y);
}


template<class Type, class Device>
void ilusv(ILU<Type, Device> ilu, Vect<Type, Device> rhs);


template<class Type, class Device>
void ilu_jacobi_mv(bool lower,
                   Type alpha, ILU_Jacobi<Type, Device> ilu,
                               Vect<Type, Device> x,
                   Type beta,  Vect<Type, Device> y) {

    const auto n = ilu.n();
    auto ilu_row_map = ilu.row_map_view();
    auto ilu_inds = ilu.inds_view();
    auto ilu_vals = ilu.vals_view();
    auto ilu_diag_inds = ilu.diag_inds_view();
    auto x_view = x.view();
    auto y_view = y.view();

    if (lower) {
        Kokkos::parallel_for(Kokkos::RangePolicy<typename Device::execution_space>(0, x.n()),
                             KOKKOS_LAMBDA(const size_t& i) {
            Type sum = x_view(i);
            auto row_start = ilu_row_map(i);
            auto row_end = ilu_diag_inds(i);

            for (int j = row_start; j < row_end; ++j) {
                auto ind = ilu_inds(j);
                sum += ilu_vals(j)*x_view(ind);
            }
            y_view(i) = beta*y_view(i) + alpha*sum;
        });
    } else {
        Kokkos::parallel_for(Kokkos::RangePolicy<typename Device::execution_space>(0, n),
                             KOKKOS_LAMBDA(const size_t& i) {
            Type sum = 0;
            auto row_start = ilu_diag_inds(i);
            auto row_end = ilu_row_map(i+1);

            for (int j = row_start; j < row_end; ++j) {
                auto ind = ilu_inds(j);
                sum += ilu_vals(j)*x_view(ind);
            }
            y_view(i) = 1.0*y_view(i) + -1.0*sum;
        });
    }
}

template<class ScalarType, class Type, class Device>
void ilu_jacobi_mv(bool lower,
                   ScalarType alpha, ILU_Jacobi<Type, Device> ilu,
                                     Vect<Type, Device> x,
                   ScalarType beta,  Vect<Type, Device> y) {
    ilu_jacobi_mv(lower, Type(alpha), ilu, x, Type(beta), y);
}

template<class Type, class Device>
void ilusv_jacobi(ILU_Jacobi<Type, Device> ilu, Vect<Type, Device> x) {
    assert(ilu.n() == x.n());

    const int steps = ilu.steps();

    auto b = ilu.temp1();
    copy(x, b);
    auto temp= ilu.temp2();

    // approximate inverse of L
    for (int i = 0; i < steps; ++i) {
        copy(b, temp);
        ilu_jacobi_mv(true, -1.0, ilu, x, 1.0, temp);
        axpy(1.0, temp, x);
    }

    copy(x, b);

    // approximate inverse of U
    for (int i = 0; i < steps; ++i) {
        copy(b, temp);
        ilu_jacobi_mv(false, -1.0, ilu, x, 0.0, temp);
        gdmv(1.0, ilu.diag_vect(), temp, 1.0, x);
    }
}

#endif // KERNELS_HPP
