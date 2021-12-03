
#include "kernels.hpp"

#include<cstdlib>

#include<mkl.h>
#include<mkl_spblas.h>

#include<Kokkos_Core.hpp>

#include "types.hpp"
#include "types_mkl.hpp"


// Functions for types_mkl

template<>
void SparseMatrix<double, MKL>::create_mkl_handles() {
    assert(mkl_sparse_d_create_csr(&mkl_handle_, SPARSE_INDEX_BASE_ZERO,
                                   m_, n_,
                                   row_map_.data(), row_map_.data()+1,
                                   inds_.data(), vals_.data())
           == SPARSE_STATUS_SUCCESS);
    mkl_desc_.type = SPARSE_MATRIX_TYPE_GENERAL;
};

template<>
void SparseMatrix<float, MKL>::create_mkl_handles() {
    assert(mkl_sparse_s_create_csr(&mkl_handle_, SPARSE_INDEX_BASE_ZERO,
                                   m_, n_,
                                   row_map_.data(), row_map_.data()+1,
                                   inds_.data(), vals_.data())
           == SPARSE_STATUS_SUCCESS);
    mkl_desc_.type = SPARSE_MATRIX_TYPE_GENERAL;
};

template<>
void ILU<double, MKL>::create_mkl_handles() {
    assert(mkl_sparse_d_create_csr(&mkl_handle_, SPARSE_INDEX_BASE_ZERO,
                                   n_, n_,
                                   row_map_.data(), row_map_.data()+1,
                                   inds_.data(), vals_.data())
           == SPARSE_STATUS_SUCCESS);

    mkl_L_desc_.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    mkl_L_desc_.mode = SPARSE_FILL_MODE_LOWER;
    mkl_L_desc_.diag = SPARSE_DIAG_UNIT;

    mkl_U_desc_.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    mkl_U_desc_.mode = SPARSE_FILL_MODE_UPPER;
    mkl_U_desc_.diag = SPARSE_DIAG_NON_UNIT;
};

template<>
void ILU<float, MKL>::create_mkl_handles() {
    assert(mkl_sparse_s_create_csr(&mkl_handle_, SPARSE_INDEX_BASE_ZERO,
                                   n_, n_,
                                   row_map_.data(), row_map_.data()+1,
                                   inds_.data(), vals_.data())
           == SPARSE_STATUS_SUCCESS);

    mkl_L_desc_.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    mkl_L_desc_.mode = SPARSE_FILL_MODE_LOWER;
    mkl_L_desc_.diag = SPARSE_DIAG_UNIT;

    mkl_U_desc_.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
    mkl_U_desc_.mode = SPARSE_FILL_MODE_UPPER;
    mkl_U_desc_.diag = SPARSE_DIAG_NON_UNIT;
};

// BLAS-1

template <>
double dot<double, MKL>(Vect<double, MKL> x, Vect<double, MKL> y) {
    assert(x.n() == y.n());
    return cblas_ddot(x.n(), x.data(), 1, y.data(), 1);
}

template <>
float dot<float, MKL>(Vect<float, MKL> x, Vect<float, MKL> y) {
    assert(x.n() == y.n());
    return cblas_sdot(x.n(), x.data(), 1, y.data(), 1);
}

template <>
void dot<double, MKL>(Vect<double, MKL> x, Vect<double, MKL> y, Scalar<double, MKL> result) {
    assert(x.n() == y.n());
    result.data()[0] = cblas_ddot(x.n(), x.data(), 1, y.data(), 1);
}

template <>
void dot<float, MKL>(Vect<float, MKL> x, Vect<float, MKL> y, Scalar<float, MKL> result) {
    assert(x.n() == y.n());
    result.data()[0] = cblas_sdot(x.n(), x.data(), 1, y.data(), 1);
}

template <>
double nrm2<double, MKL>(Vect<double, MKL> x) {
    return cblas_dnrm2(x.n(), x.data(), 1);
}

template <>
float nrm2<float, MKL>(Vect<float, MKL> x) {
    return cblas_snrm2(x.n(), x.data(), 1);
}

template <>
void nrm2<double, MKL>(Vect<double, MKL> x, Scalar<double, MKL> result) {
    result.data()[0] = cblas_dnrm2(x.n(), x.data(), 1);
}

template <>
void nrm2<float, MKL>(Vect<float, MKL> x, Scalar<float, MKL> result) {
    result.data()[0] = cblas_snrm2(x.n(), x.data(), 1);
}


template <>
void axpy<double, MKL>(double alpha, Vect<double, MKL> x, Vect<double, MKL> y) {
    assert(x.n() == y.n());
    return cblas_daxpy(x.n(), alpha, x.data(), 1, y.data(), 1);
}

template <>
void axpy<float, MKL>(float alpha, Vect<float, MKL> x, Vect<float, MKL> y) {
    assert(x.n() == y.n());
    return cblas_saxpy(x.n(), alpha, x.data(), 1, y.data(), 1);
}

template <>
void axpy<double, MKL>(Scalar<double, MKL> alpha, Vect<double, MKL> x, Vect<double, MKL> y) {
    assert(x.n() == y.n());
    return cblas_daxpy(x.n(), alpha.data()[0], x.data(), 1, y.data(), 1);
}

template <>
void axpy<float, MKL>(Scalar<float, MKL> alpha, Vect<float, MKL> x, Vect<float, MKL> y) {
    assert(x.n() == y.n());
    return cblas_saxpy(x.n(), alpha.data()[0], x.data(), 1, y.data(), 1);
}


template <>
void naxpy<double, MKL>(Scalar<double, MKL> alpha, Vect<double, MKL> x, Vect<double, MKL> y) {
    assert(x.n() == y.n());
    return cblas_daxpy(x.n(), -alpha.data()[0], x.data(), 1, y.data(), 1);
}

template <>
void naxpy<float, MKL>(Scalar<float, MKL> alpha, Vect<float, MKL> x, Vect<float, MKL> y) {
    assert(x.n() == y.n());
    return cblas_saxpy(x.n(), -alpha.data()[0], x.data(), 1, y.data(), 1);
}

template <>
void scal<double, MKL>(double alpha, Vect<double, MKL> x) {
    return cblas_dscal(x.n(), alpha, x.data(), 1);
}

template <>
void scal<float, MKL>(float alpha, Vect<float, MKL> x) {
    return cblas_sscal(x.n(), alpha, x.data(), 1);
}

template <>
void scal<double, MKL>(Scalar<double, MKL> alpha, Vect<double, MKL> x, Vect<double, MKL> y) {
    assert(x.n() == y.n());
    copy(x, y);
    return cblas_dscal(x.n(), alpha.data()[0], y.data(), 1);
}

template <>
void scal<float, MKL>(Scalar<float, MKL> alpha, Vect<float, MKL> x, Vect<float, MKL> y) {
    assert(x.n() == y.n());
    copy(x, y);
    return cblas_sscal(x.n(), alpha.data()[0], y.data(), 1);
}

template <>
void scal<double, MKL>(double alpha, Vect<double, MKL> x, Vect<double, MKL> y) {
    assert(x.n() == y.n());
    copy(x, y);
    return cblas_dscal(x.n(), alpha, y.data(), 1);
}

template <>
void scal<float, MKL>(float alpha, Vect<float, MKL> x, Vect<float, MKL> y) {
    assert(x.n() == y.n());
    copy(x, y);
    return cblas_sscal(x.n(), alpha, y.data(), 1);
}

template <>
void scal<double, MKL>(Scalar<double, MKL> alpha, Scalar<double, MKL> x, Scalar<double, MKL> y) {
    y.data()[0] = alpha.data()[0]*x.data()[0];
}

template <>
void scal<float, MKL>(Scalar<float, MKL> alpha, Scalar<float, MKL> x, Scalar<float, MKL> y) {
    y.data()[0] = alpha.data()[0]*x.data()[0];
}

template <>
void scal<double, MKL>(double alpha, Scalar<double, MKL> x, Scalar<double, MKL> y) {
    y.data()[0] = alpha*x.data()[0];
}

template <>
void scal<float, MKL>(float alpha, Scalar<float, MKL> x, Scalar<float, MKL> y) {
    y.data()[0] = alpha*x.data()[0];
}


template <>
void rotg<double, MKL>(Scalar<double, MKL> a, Scalar<double, MKL> b,
                       Scalar<double, MKL> c, Scalar<double, MKL> s) {
    cblas_drotg(a.data(), b.data(), c.data(), s.data());
    b.data()[0] = 0.0;
}

template <>
void rotg<float, MKL>(Scalar<float, MKL> a, Scalar<float, MKL> b,
                      Scalar<float, MKL> c, Scalar<float, MKL> s) {
    cblas_srotg(a.data(), b.data(), c.data(), s.data());
    b.data()[0] = 0.0;
}

template <>
void rot<double, MKL>(Scalar<double, MKL> a, Scalar<double, MKL> b,
                      Scalar<double, MKL> c, Scalar<double, MKL> s) {
    cblas_drot(1, a.data(), 1, b.data(), 1, *(c.data()), *(s.data()));
}

template <>
void rot<float, MKL>(Scalar<float, MKL> a, Scalar<float, MKL> b,
                     Scalar<float, MKL> c, Scalar<float, MKL> s) {
    cblas_srot(1, a.data(), 1, b.data(), 1, *(c.data()), *(s.data()));
}

template<>
void rot<double, MKL>(Vect<double, MKL> a, Vect<double, MKL> c, Vect<double, MKL> s) {
    int k = c.n();
    auto a_data = a.data();
    auto c_data = c.data();
    auto s_data = s.data();
    for (size_t j = 0; j < k; j++) {
        cblas_drot(1, a_data+j, 1, a_data+j+1, 1, c_data[j], s_data[j]);
    }
}

template<>
void rot<float, MKL>(Vect<float, MKL> a, Vect<float, MKL> c, Vect<float, MKL> s) {
    int k = c.n();
    auto a_data = a.data();
    auto c_data = c.data();
    auto s_data = s.data();
    for (size_t j = 0; j < k; j++) {
        cblas_srot(1, a_data+j, 1, a_data+j+1, 1, c_data[j], s_data[j]);
    }
}

// BLAS-2

template <>
void gemv<double, MKL>(double alpha, MultiVect<double, MKL> matrix, Vect<double, MKL> x, double beta, Vect<double, MKL> y) {
    assert(matrix.ncols() == x.n());
    assert(matrix.nrows() == y.n());

    const CBLAS_TRANSPOSE trans = matrix.transposed() ? CblasTrans : CblasNoTrans;

    cblas_dgemv(CblasColMajor, trans,
                matrix.nrows_base(), matrix.ncols_base(), alpha, matrix.data(), matrix.stride(),
                x.data(), 1,
                beta, y.data(), 1);
}

template <>
void gemv<float, MKL>(float alpha, MultiVect<float, MKL> matrix, Vect<float, MKL> x, float beta, Vect<float, MKL> y) {
    assert(matrix.ncols() == x.n());
    assert(matrix.nrows() == y.n());

    const CBLAS_TRANSPOSE trans = matrix.transposed() ? CblasTrans : CblasNoTrans;

    cblas_sgemv(CblasColMajor, trans,
                matrix.nrows_base(), matrix.ncols_base(), alpha, matrix.data(), matrix.stride(),
                x.data(), 1,
                beta, y.data(), 1);
}


template <>
void trsv<double, MKL>(const char* upper, MultiVect<double, MKL> matrix, Vect<double, MKL> x) {
    const auto nrows = matrix.nrows();
    const auto ncols = matrix.ncols();

    assert(ncols == nrows);
    assert(ncols == x.n());

    const auto uplo = 'U' == *upper ? CblasUpper : CblasLower;
    const auto trans = matrix.transposed() ? CblasTrans : CblasNoTrans;

    cblas_dtrsv(CblasColMajor, uplo, trans, CblasNonUnit,
                ncols, matrix.data(), matrix.stride(),
                       x.data(), 1);
}

template <>
void trsv<float, MKL>(const char* upper, MultiVect<float, MKL> matrix, Vect<float, MKL> x) {
    const auto nrows = matrix.nrows();
    const auto ncols = matrix.ncols();

    assert(ncols == nrows);
    assert(ncols == x.n());

    const auto uplo = 'U' == *upper ? CblasUpper : CblasLower;
    const auto trans = matrix.transposed() ? CblasTrans : CblasNoTrans;

    cblas_strsv(CblasColMajor, uplo, trans, CblasNonUnit,
                ncols, matrix.data(), matrix.stride(),
                       x.data(), 1);
}


// Sparse

template <>
void spmv<double, MKL>(double alpha, SparseMatrix<double, MKL> matrix, Vect<double, MKL> x, double beta, Vect<double, MKL> y) {
    const auto nrows = matrix.nrows();
    const auto ncols = matrix.ncols();

    assert(ncols == x.n());
    assert(nrows == y.n());

    assert(mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                           alpha, matrix.mkl_handle(), matrix.mkl_desc(), x.data(),
                           beta, y.data())
            == SPARSE_STATUS_SUCCESS);
}

template <>
void spmv<float, MKL>(float alpha, SparseMatrix<float, MKL> matrix, Vect<float, MKL> x, float beta, Vect<float, MKL> y) {
    const auto nrows = matrix.nrows();
    const auto ncols = matrix.ncols();

    assert(ncols == x.n());
    assert(nrows == y.n());

    assert(mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                           alpha, matrix.mkl_handle(), matrix.mkl_desc(), x.data(),
                           beta, y.data())
            == SPARSE_STATUS_SUCCESS);
}


template <>
void ilusv<double, MKL>(ILU<double, MKL> ilu, Vect<double, MKL> x) {
    assert(ilu.n() == x.n());
    const auto n = ilu.n();

    assert(mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE,
                             1.0, ilu.mkl_handle(), ilu.mkl_L_desc(),
                             x.data(), x.data())
           == SPARSE_STATUS_SUCCESS);

    assert(mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE,
                             1.0, ilu.mkl_handle(), ilu.mkl_U_desc(),
                             x.data(), x.data())
           == SPARSE_STATUS_SUCCESS);
}

template <>
void ilusv<float, MKL>(ILU<float, MKL> ilu, Vect<float, MKL> x) {
    assert(ilu.n() == x.n());

    assert(mkl_sparse_s_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0,
                             ilu.mkl_handle(), ilu.mkl_L_desc(),
                             x.data(), x.data())
           == SPARSE_STATUS_SUCCESS);
    assert(mkl_sparse_s_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0,
                            ilu.mkl_handle(), ilu.mkl_U_desc(),
                            x.data(), x.data())
           == SPARSE_STATUS_SUCCESS);
}


template<>
void ilu_jacobi_mv<double, MKL>(bool lower,
                                double alpha, ILU_Jacobi<double, MKL> ilu,
                                              Vect<double, MKL> x,
                                double beta,  Vect<double, MKL> y) {
    const auto n = ilu.n();
    const auto nnz = ilu.nnz();
    auto desc = lower ? ilu.handles().L_desc() : ilu.handles().U_desc();

    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                    alpha, ilu.handles().matrix(), desc,
                            x.data(),
                    beta,  y.data());
}

template<>
void ilu_jacobi_mv<float, MKL>(bool lower,
                               float alpha, ILU_Jacobi<float, MKL> ilu,
                                            Vect<float, MKL> x,
                               float beta,  Vect<float, MKL> y) {
    const auto n = ilu.n();
    const auto nnz = ilu.nnz();
    auto desc = lower ? ilu.handles().L_desc() : ilu.handles().U_desc();

    mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                    alpha, ilu.handles().matrix(), desc,
                            x.data(),
                    beta,  y.data());
}

template<class Type>
inline ILU<Type, MKL> ilu0_impl(SparseMatrix<double, MKL> A) {

    typedef int ordinal_t;
    typedef int offset_t;

    double alpha = 0;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<typename MKL::execution_space>(0, A.nrows()),
                            KOKKOS_LAMBDA(const ordinal_t& i, double& update) {
        offset_t k = A.row_map(i);
        const offset_t end = A.row_map(i+1);
        double sum = 0;
        for (; k < end; k++) {
            sum += std::fabs(A.vals(k));
        }
        if (update < sum) {
            update = sum;
        }
    }, Kokkos::Max<double>(alpha));

    alpha *= std::numeric_limits<Type>::epsilon();

    const ordinal_t n = A.nrows();
    const offset_t nnz = A.row_map(n);

    typedef typename Kokkos::View<double*, typename MKL::memory_space> values_t;
    typedef typename Kokkos::View<ordinal_t*, typename MKL::memory_space> index_t;

    const auto row_map = A.row_map_;
    const auto inds = A.inds_;
    values_t vals ("ilu::vals", nnz);
    Kokkos::deep_copy(vals, A.vals_);
    index_t diag_inds ("ilu::diags", n);

    for (ordinal_t i = 1; i < n; i++) { // row
        const auto rowStart = row_map(i);
        const auto rowEnd = row_map(i+1);

        for (offset_t k_ind = rowStart; inds(k_ind) < i; ++k_ind) {
            const ordinal_t k = inds(k_ind); // column

            ordinal_t prev_ind = diag_inds(k);
            const ordinal_t prev_end = row_map(k+1);

            const double factor = vals(k_ind) / vals(prev_ind);
            vals(k_ind) = factor;

            prev_ind += 1;
            for (offset_t j_ind = k_ind + 1; j_ind < rowEnd && prev_ind < prev_end; ) {
                if (inds(prev_ind) < inds(j_ind)) {
                    ++prev_ind;
                } else if (inds(prev_ind) > inds(j_ind)) {
                    ++j_ind;
                } else { // inds(prev_ind) == inds(j_ind)
                    vals(j_ind) -= factor*vals(prev_ind);
                    ++prev_ind;
                    ++j_ind;
                }
            }
        }

        if (vals(diag_inds(i)) >= 0) {
            if (vals(diag_inds(i)) < alpha) {
                vals(diag_inds(i)) = alpha;
            }
        } else {
            if (vals(diag_inds(i)) > -alpha) {
                vals(diag_inds(i)) = -alpha;
            }
        }
    }

    typedef typename Kokkos::View<Type*, typename MKL::memory_space> values_type;
    values_type vals_type ("ilu::vals", nnz);
    Kokkos::parallel_for(Kokkos::RangePolicy<typename MKL::execution_space>(0, vals.extent(0)),
                         KOKKOS_LAMBDA(const size_t& i) {
        vals_type(i) = vals(i);
    });

    return ILU<Type, MKL>(n, row_map, inds, vals_type);
}

template <>
ILU<double, MKL> ilu0<double, MKL>(SparseMatrix<double, MKL> matrix) {
    return ilu0_impl<double>(matrix);
}

template <>
ILU<float, MKL> ilu0<float, MKL>(SparseMatrix<double, MKL> matrix) {
    return ilu0_impl<float>(matrix);
}
