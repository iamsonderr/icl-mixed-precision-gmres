
#include "kernels.hpp"

#include<cublas_v2.h>
#include<cusparse_v2.h>

#include<Kokkos_Core.hpp>

#include "types_cuda.hpp"

// Functions for types_cuda

template<>
void ILU<double, Cuda>::create_cuda_handles() {
    int pBufferSize, pBufferSize_L, pBufferSize_U;
    void* pBuffer;

    cusparseMatDescr_t cuda_L_desc;
    csrsv2Info_t cuda_L_info;
    cusparseMatDescr_t cuda_U_desc;
    csrsv2Info_t cuda_U_info;

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    cusparseCreateCsrsv2Info(&cuda_L_info);
    cusparseCreateCsrsv2Info(&cuda_U_info);

    cusparseCreateMatDescr(&cuda_L_desc);
    cusparseSetMatIndexBase(cuda_L_desc, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(cuda_L_desc, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(cuda_L_desc, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(cuda_L_desc, CUSPARSE_DIAG_TYPE_UNIT);

    cusparseCreateMatDescr(&cuda_U_desc);
    cusparseSetMatIndexBase(cuda_U_desc, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(cuda_U_desc, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(cuda_U_desc, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(cuda_U_desc, CUSPARSE_DIAG_TYPE_NON_UNIT);


    cusparseDcsrsv2_bufferSize(handles.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n_, nnz_,
        cuda_L_desc, vals_data(), row_map_data(), inds_data(), cuda_L_info, &pBufferSize_L);
    cusparseDcsrsv2_bufferSize(handles.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n_, nnz_,
        cuda_U_desc, vals_data(), row_map_data(), inds_data(), cuda_U_info, &pBufferSize_U);
    pBufferSize = max(pBufferSize_L, pBufferSize_U);
    cudaMalloc((void**)&pBuffer, pBufferSize);

    cusparseDcsrsv2_analysis(handles.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n_, nnz_, cuda_L_desc,
        vals_data(), row_map_data(), inds_data(),
        cuda_L_info, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);

    cusparseDcsrsv2_analysis(handles.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n_, nnz_, cuda_U_desc,
        vals_data(), row_map_data(), inds_data(),
        cuda_U_info, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);

    cudaFree(pBuffer);

    handles_ = std::make_shared<ILU_handles>(cuda_L_desc, cuda_L_info, cuda_U_desc, cuda_U_info);
};

template<>
void ILU<float, Cuda>::create_cuda_handles() {
    int pBufferSize, pBufferSize_L, pBufferSize_U;
    void* pBuffer;

    cusparseMatDescr_t cuda_L_desc;
    csrsv2Info_t cuda_L_info;
    cusparseMatDescr_t cuda_U_desc;
    csrsv2Info_t cuda_U_info;

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    cusparseCreateCsrsv2Info(&cuda_L_info);
    cusparseCreateCsrsv2Info(&cuda_U_info);

    cusparseCreateMatDescr(&cuda_L_desc);
    cusparseSetMatIndexBase(cuda_L_desc, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(cuda_L_desc, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(cuda_L_desc, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(cuda_L_desc, CUSPARSE_DIAG_TYPE_UNIT);

    cusparseCreateMatDescr(&cuda_U_desc);
    cusparseSetMatIndexBase(cuda_U_desc, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(cuda_U_desc, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(cuda_U_desc, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(cuda_U_desc, CUSPARSE_DIAG_TYPE_NON_UNIT);


    cusparseScsrsv2_bufferSize(handles.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n_, nnz_,
        cuda_L_desc, vals_data(), row_map_data(), inds_data(), cuda_L_info, &pBufferSize_L);
    cusparseScsrsv2_bufferSize(handles.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n_, nnz_,
        cuda_U_desc, vals_data(), row_map_data(), inds_data(), cuda_U_info, &pBufferSize_U);
    pBufferSize = max(pBufferSize_L, pBufferSize_U);
    cudaMalloc((void**)&pBuffer, pBufferSize);

    cusparseScsrsv2_analysis(handles.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n_, nnz_, cuda_L_desc,
        vals_data(), row_map_data(), inds_data(),
        cuda_L_info, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);

    cusparseScsrsv2_analysis(handles.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n_, nnz_, cuda_U_desc,
        vals_data(), row_map_data(), inds_data(),
        cuda_U_info, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);

    cudaFree(pBuffer);

    handles_ = std::make_shared<ILU_handles>(cuda_L_desc, cuda_L_info, cuda_U_desc, cuda_U_info);
};

// BLAS-1

template <>
double dot<double, Cuda>(Vect<double, Cuda> x, Vect<double, Cuda> y) {
    assert(x.n() == y.n());

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    double result;
    cublasDdot(handles.cublas_handle, x.n(),
               x.data(), 1,
               y.data(), 1,
               &result);
    return result;
}

template <>
float dot<float, Cuda>(Vect<float, Cuda> x, Vect<float, Cuda> y) {
    assert(x.n() == y.n());

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    float result;
    cublasSdot(handles.cublas_handle, x.n(),
               x.data(), 1,
               y.data(), 1,
               &result);
    return result;
}

template <>
void dot<double, Cuda>(Vect<double, Cuda> x, Vect<double, Cuda> y, Scalar<double, Cuda> result) {
    assert(x.n() == y.n());

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    cublasSetPointerMode(handles.cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
    cublasDdot(handles.cublas_handle, x.n(),
               x.data(), 1,
               y.data(), 1,
               result.data());
    cublasSetPointerMode(handles.cublas_handle, CUBLAS_POINTER_MODE_HOST);
}

template <>
void dot<float, Cuda>(Vect<float, Cuda> x, Vect<float, Cuda> y, Scalar<float, Cuda> result) {
    assert(x.n() == y.n());

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    cublasSetPointerMode(handles.cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
    cublasSdot(handles.cublas_handle, x.n(),
               x.data(), 1,
               y.data(), 1,
               result.data());
    cublasSetPointerMode(handles.cublas_handle, CUBLAS_POINTER_MODE_HOST);
}

template <>
double nrm2<double, Cuda>(Vect<double, Cuda> x) {
    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    double result;
    cublasDnrm2(handles.cublas_handle, x.n(),
                x.data(), 1,
                &result);
    return result;
}

template <>
float nrm2<float, Cuda>(Vect<float, Cuda> x) {
    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    float result;
    cublasSnrm2(handles.cublas_handle, x.n(),
                x.data(), 1,
                &result);
    return result;
}

template <>
void nrm2<double, Cuda>(Vect<double, Cuda> x, Scalar<double, Cuda> result) {
    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    cublasSetPointerMode(handles.cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
    cublasDnrm2(handles.cublas_handle, x.n(),
                x.data(), 1,
                result.data());
    cublasSetPointerMode(handles.cublas_handle, CUBLAS_POINTER_MODE_HOST);
}

template <>
void nrm2<float, Cuda>(Vect<float, Cuda> x, Scalar<float, Cuda> result) {
    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    cublasSetPointerMode(handles.cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
    cublasSnrm2(handles.cublas_handle, x.n(),
                x.data(), 1,
                result.data());
    cublasSetPointerMode(handles.cublas_handle, CUBLAS_POINTER_MODE_HOST);
}


template <>
void axpy<double, Cuda>(double alpha, Vect<double, Cuda> x, Vect<double, Cuda> y) {
    assert(x.n() == y.n());

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    cublasDaxpy(handles.cublas_handle, x.n(),
                &alpha,
                x.data(), 1,
                y.data(), 1);
}

template <>
void axpy<float, Cuda>(float alpha, Vect<float, Cuda> x, Vect<float, Cuda> y) {
    assert(x.n() == y.n());

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    cublasSaxpy(handles.cublas_handle, x.n(),
                &alpha,
                x.data(), 1,
                y.data(), 1);
}

template <>
void axpy<double, Cuda>(Scalar<double, Cuda> alpha, Vect<double, Cuda> x, Vect<double, Cuda> y) {
    assert(x.n() == y.n());

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    cublasSetPointerMode(handles.cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
    cublasDaxpy(handles.cublas_handle, x.n(),
                alpha.data(),
                x.data(), 1,
                y.data(), 1);
    cublasSetPointerMode(handles.cublas_handle, CUBLAS_POINTER_MODE_HOST);
}

template <>
void axpy<float, Cuda>(Scalar<float, Cuda> alpha, Vect<float, Cuda> x, Vect<float, Cuda> y) {
    assert(x.n() == y.n());

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    cublasSetPointerMode(handles.cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
    cublasSaxpy(handles.cublas_handle, x.n(),
                alpha.data(),
                x.data(), 1,
                y.data(), 1);
    cublasSetPointerMode(handles.cublas_handle, CUBLAS_POINTER_MODE_HOST);
}

template <>
void naxpy<double, Cuda>(Scalar<double, Cuda> alpha, Vect<double, Cuda> x, Vect<double, Cuda> y) {
    assert(x.n() == y.n());

    auto x_view = x.view();
    auto y_view = y.view();
    auto alpha_view = alpha.view();
    Kokkos::parallel_for(Kokkos::RangePolicy<typename Cuda::execution_space>(0, x.n()),
    					 KOKKOS_LAMBDA(const size_t& i) {
    	y_view(i) -= alpha_view()*x_view(i);
    });
}

template <>
void naxpy<float, Cuda>(Scalar<float, Cuda> alpha, Vect<float, Cuda> x, Vect<float, Cuda> y) {
    assert(x.n() == y.n());

    auto x_view = x.view();
    auto y_view = y.view();
    auto alpha_view = alpha.view();
    Kokkos::parallel_for(Kokkos::RangePolicy<typename Cuda::execution_space>(0, x.n()),
    					 KOKKOS_LAMBDA(const size_t& i) {
    	y_view(i) -= alpha_view()*x_view(i);
    });
}


template <>
void scal<double, Cuda>(double alpha, Vect<double, Cuda> x) {
    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    cublasDscal(handles.cublas_handle, x.n(),
                &alpha,
                x.data(), 1);
}

template <>
void scal<float, Cuda>(float alpha, Vect<float, Cuda> x) {
    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    cublasSscal(handles.cublas_handle, x.n(),
                &alpha,
                x.data(), 1);
}

template <>
void scal<double, Cuda>(double alpha, Vect<double, Cuda> x, Vect<double, Cuda> y) {
    assert(x.n() == y.n());

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    copy(x, y);
    cublasDscal(handles.cublas_handle, x.n(),
                &alpha,
                y.data(), 1);
}

template <>
void scal<float, Cuda>(float alpha, Vect<float, Cuda> x, Vect<float, Cuda> y) {
    assert(x.n() == y.n());

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    copy(x, y);
    cublasSscal(handles.cublas_handle, x.n(),
                &alpha,
                y.data(), 1);
}

template <>
void scal<double, Cuda>(Scalar<double, Cuda> alpha, Vect<double, Cuda> x, Vect<double, Cuda> y) {
    assert(x.n() == y.n());

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    copy(x, y);
    cublasSetPointerMode(handles.cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
    cublasDscal(handles.cublas_handle, x.n(),
                alpha.data(),
                y.data(), 1);
    cublasSetPointerMode(handles.cublas_handle, CUBLAS_POINTER_MODE_HOST);
}

template <>
void scal<float, Cuda>(Scalar<float, Cuda> alpha, Vect<float, Cuda> x, Vect<float, Cuda> y) {
    assert(x.n() == y.n());

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    copy(x, y);
    cublasSetPointerMode(handles.cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
    cublasSscal(handles.cublas_handle, x.n(),
                alpha.data(),
                y.data(), 1);
    cublasSetPointerMode(handles.cublas_handle, CUBLAS_POINTER_MODE_HOST);
}


template <>
void scal<double, Cuda>(const double alpha, Scalar<double, Cuda> x, Scalar<double, Cuda> y) {
    // CudaLibSingleton& handles = CudaLibSingleton::singleton();
    //
    //copy(x, y);
    //cublasDscal(handles.cublas_handle, 1,
    //            &alpha,
    //            y.data(), 1);
    auto x_view = x.view();
    auto y_view = y.view();
    Kokkos::parallel_for(Kokkos::RangePolicy<typename Cuda::execution_space>(0, 1),
                         KOKKOS_LAMBDA(const size_t& i) {
        y_view() = alpha*x_view();
    });
}

template <>
void scal<float, Cuda>(float alpha, Scalar<float, Cuda> x, Scalar<float, Cuda> y) {
    // CudaLibSingleton& handles = CudaLibSingleton::singleton();
    //
    // copy(x, y);
    // cublasSscal(handles.cublas_handle, 1,
    //             &alpha,
    //             y.data(), 1);
    auto x_view = x.view();
    auto y_view = y.view();
    Kokkos::parallel_for(Kokkos::RangePolicy<typename Cuda::execution_space>(0, 1),
                         KOKKOS_LAMBDA(const size_t& i) {
        y_view() = alpha*x_view();
    });
}

template <>
void rotg<double, Cuda>(Scalar<double, Cuda> a, Scalar<double, Cuda> b,
                        Scalar<double, Cuda> c, Scalar<double, Cuda> s) {

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    cublasSetPointerMode(handles.cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
    cublasDrotg(handles.cublas_handle,
                a.data(), b.data(),
                c.data(), s.data());
    fill(0.0, b);
    cublasSetPointerMode(handles.cublas_handle, CUBLAS_POINTER_MODE_HOST);
}

template <>
void rotg<float, Cuda>(Scalar<float, Cuda> a, Scalar<float, Cuda> b,
                       Scalar<float, Cuda> c, Scalar<float, Cuda> s) {

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    cublasSetPointerMode(handles.cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
    cublasSrotg(handles.cublas_handle,
                a.data(), b.data(),
                c.data(), s.data());
    fill(0.0, b);
    cublasSetPointerMode(handles.cublas_handle, CUBLAS_POINTER_MODE_HOST);
}

template <>
void rot<double, Cuda>(Scalar<double, Cuda> a, Scalar<double, Cuda> b,
                       Scalar<double, Cuda> c, Scalar<double, Cuda> s) {

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    cublasSetPointerMode(handles.cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
    cublasDrot(handles.cublas_handle, 1,
               a.data(), 1, b.data(), 1,
               c.data(), s.data());
    cublasSetPointerMode(handles.cublas_handle, CUBLAS_POINTER_MODE_HOST);
}

template <>
void rot<float, Cuda>(Scalar<float, Cuda> a, Scalar<float, Cuda> b,
                      Scalar<float, Cuda> c, Scalar<float, Cuda> s) {

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    cublasSetPointerMode(handles.cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
    cublasSrot(handles.cublas_handle, 1,
               a.data(), 1, b.data(), 1,
               c.data(), s.data());
    cublasSetPointerMode(handles.cublas_handle, CUBLAS_POINTER_MODE_HOST);
}

template <>
void rot<double, Cuda>(Vect<double, Cuda> a, Vect<double, Cuda> c, Vect<double, Cuda> s) {

    const int k = c.n();
    // for (size_t j = 0; j < k; j++) {
    //     rot(a(j), a(j+1), c(j), s(j));
    // }
    auto a_view = a.view();
    auto c_view = c.view();
    auto s_view = s.view();

    Kokkos::parallel_for(Kokkos::RangePolicy<typename Cuda::execution_space>(0, 1),
                         KOKKOS_LAMBDA(const size_t) {
         for (size_t j = 0; j < k; j++) {
            double a1 = a_view(j);
            double a2 = a_view(j+1);
            double c = c_view(j);
            double s = s_view(j);
            a_view(j)  =  c*a1 + s*a2;
            a_view(j+1) = -s*a1 + c*a2;
        }
    });
}

template <>
void rot<float, Cuda>(Vect<float, Cuda> a, Vect<float, Cuda> c, Vect<float, Cuda> s) {

    const int k = c.n();
    // for (size_t j = 0; j < k; j++) {
    //     rot(a(j), a(j+1), c(j), s(j));
    // }
    auto a_view = a.view();
    auto c_view = c.view();
    auto s_view = s.view();

    Kokkos::parallel_for(Kokkos::RangePolicy<typename Cuda::execution_space>(0, 1),
                         KOKKOS_LAMBDA(const size_t) {
         for (size_t j = 0; j < k; j++) {
            float a1 = a_view(j);
            float a2 = a_view(j+1);
            float c = c_view(j);
            float s = s_view(j);
            a_view(j)  =  c*a1 + s*a2;
            a_view(j+1) = -s*a1 + c*a2;
        }
    });
}

// BLAS-2


template<>
void gemv<double, Cuda>(double alpha, MultiVect<double, Cuda> matrix, Vect<double, Cuda> x, double beta, Vect<double, Cuda> y) {
    const auto nrows = matrix.nrows();
    const auto ncols = matrix.ncols();

    assert(ncols == x.n());
    assert(nrows == y.n());

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    const auto trans = matrix.transposed() ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasDgemv(handles.cublas_handle, trans,
                matrix.nrows_base(), matrix.ncols_base(),
                &alpha, matrix.data(), matrix.stride(),
                        x.data(), 1,
                &beta,  y.data(), 1);
}

template<>
void gemv<float, Cuda>(float alpha, MultiVect<float, Cuda> matrix, Vect<float, Cuda> x, float beta, Vect<float, Cuda> y) {
    const auto nrows = matrix.nrows();
    const auto ncols = matrix.ncols();

    assert(ncols == x.n());
    assert(nrows == y.n());

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    const auto trans = matrix.transposed() ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasSgemv(handles.cublas_handle, trans,
                matrix.nrows_base(), matrix.ncols_base(),
                &alpha, matrix.data(), matrix.stride(),
                        x.data(), 1,
                &beta,  y.data(), 1);
}


template<>
void trsv<double, Cuda>(const char* upper, MultiVect<double, Cuda> matrix, Vect<double, Cuda> x) {
    const auto nrows = matrix.nrows();
    const auto ncols = matrix.ncols();

    assert(ncols == nrows);
    assert(ncols == x.n());

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    const auto uplo = 'U' == *upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    const auto trans = matrix.transposed() ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasDtrsv(handles.cublas_handle, uplo, trans, CUBLAS_DIAG_NON_UNIT,
                nrows, matrix.data(), matrix.stride(),
                       x.data(), 1);
}

template<>
void trsv<float, Cuda>(const char* upper, MultiVect<float, Cuda> matrix, Vect<float, Cuda> x) {
    const auto nrows = matrix.nrows();
    const auto ncols = matrix.ncols();

    assert(ncols == nrows);
    assert(ncols == x.n());

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    const auto uplo = 'U' == *upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    const auto trans = matrix.transposed() ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasStrsv(handles.cublas_handle, uplo, trans, CUBLAS_DIAG_NON_UNIT,
                nrows, matrix.data(), matrix.stride(),
                       x.data(), 1);
}


// Sparse
template<>
void spmv<double, Cuda>(double alpha, SparseMatrix<double, Cuda> matrix, Vect<double, Cuda> x, double beta, Vect<double, Cuda> y) {
    const auto nrows = matrix.nrows();
    const auto ncols = matrix.ncols();
    const auto nnz = matrix.nnz();

    assert(ncols == x.n());
    assert(nrows == y.n());

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    cusparseOperation_t op = matrix.is_transposed() ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;

    cusparseDcsrmv(handles.cusparse_handle, op,
                   nrows, ncols, nnz,
                   &alpha, matrix.desc(), matrix.vals_data(), matrix.row_map_data(), matrix.inds_data(),
                   x.data(),
                   &beta, y.data());
}

template<>
void spmv<float, Cuda>(float alpha, SparseMatrix<float, Cuda> matrix, Vect<float, Cuda> x, float beta, Vect<float, Cuda> y) {
    const auto nrows = matrix.nrows();
    const auto ncols = matrix.ncols();
    const auto nnz = matrix.nnz();

    assert(ncols == x.n());
    assert(nrows == y.n());

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    cusparseOperation_t op = matrix.is_transposed() ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;

    cusparseScsrmv(handles.cusparse_handle, op,
                   nrows, ncols, nnz,
                   &alpha, matrix.desc(), matrix.vals_data(), matrix.row_map_data(), matrix.inds_data(),
                   x.data(),
                   &beta, y.data());
}


template<>
void ilusv<double, Cuda>(ILU<double, Cuda> ilu, Vect<double, Cuda> x) {

    const double one = 1.0;

    assert(ilu.n() == x.n());

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    // Allocate buffer
    int pBufferSize_L;
    int pBufferSize_U;
    cusparseDcsrsv2_bufferSize(handles.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            ilu.n(), ilu.nnz(), ilu.cuda_L_desc(),
            ilu.vals_data(), ilu.row_map_data(), ilu.inds_data(), ilu.cuda_L_info(),
            &pBufferSize_L);
    cusparseDcsrsv2_bufferSize(handles.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            ilu.n(), ilu.nnz(), ilu.cuda_U_desc(),
            ilu.vals_data(), ilu.row_map_data(), ilu.inds_data(), ilu.cuda_U_info(),
            &pBufferSize_U);
    int pBufferSize = max(pBufferSize_L, pBufferSize_U);
    void* pBuffer;
    cudaMalloc((void**)&pBuffer, pBufferSize);


    // Forward solve
    cusparseDcsrsv2_solve(handles.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        ilu.n(), ilu.nnz(), &one, ilu.cuda_L_desc(),
        ilu.vals_data(), ilu.row_map_data(), ilu.inds_data(), ilu.cuda_L_info(),
        x.data(), x.data(), CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);

    // Backwards solve
    cusparseDcsrsv2_solve(handles.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        ilu.n(), ilu.nnz(), &one, ilu.cuda_U_desc(),
        ilu.vals_data(), ilu.row_map_data(), ilu.inds_data(), ilu.cuda_U_info(),
        x.data(), x.data(), CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);

    cudaFree(pBuffer);
}

template<>
void ilusv<float, Cuda>(ILU<float, Cuda> ilu, Vect<float, Cuda> x) {

    const float one = 1.0;

    assert(ilu.n() == x.n());

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    // Allocate buffer
    int pBufferSize_L;
    int pBufferSize_U;
    cusparseScsrsv2_bufferSize(handles.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            ilu.n(), ilu.nnz(), ilu.cuda_L_desc(),
            ilu.vals_data(), ilu.row_map_data(), ilu.inds_data(), ilu.cuda_L_info(),
            &pBufferSize_L);
    cusparseScsrsv2_bufferSize(handles.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            ilu.n(), ilu.nnz(), ilu.cuda_U_desc(),
            ilu.vals_data(), ilu.row_map_data(), ilu.inds_data(), ilu.cuda_U_info(),
            &pBufferSize_U);
    int pBufferSize = max(pBufferSize_L, pBufferSize_U);
    void* pBuffer;
    cudaMalloc((void**)&pBuffer, pBufferSize);


    // Forward solve
    cusparseScsrsv2_solve(handles.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        ilu.n(), ilu.nnz(), &one, ilu.cuda_L_desc(),
        ilu.vals_data(), ilu.row_map_data(), ilu.inds_data(), ilu.cuda_L_info(),
        x.data(), x.data(), CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);

    // Backwards solve
    cusparseScsrsv2_solve(handles.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        ilu.n(), ilu.nnz(), &one, ilu.cuda_U_desc(),
        ilu.vals_data(), ilu.row_map_data(), ilu.inds_data(), ilu.cuda_U_info(),
        x.data(), x.data(), CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);

    cudaFree(pBuffer);
}

template<class Type>
Kokkos::View<Type*, typename Cuda::memory_space>
type_convert(Kokkos::View<double*, typename Cuda::memory_space> vals) {
    Kokkos::View<Type*, typename Cuda::memory_space> vals_type = Kokkos::View<Type*, typename Cuda::memory_space>("ilu::vals", vals.extent(0));
    Kokkos::parallel_for(Kokkos::RangePolicy<typename Cuda::execution_space>(0, vals.extent(0)),
                         KOKKOS_LAMBDA(const size_t& i) {
        vals_type(i) = vals(i);
    });
    return vals_type;
}

template<>
Kokkos::View<double*, typename Cuda::memory_space>
type_convert<double>(Kokkos::View<double*, typename Cuda::memory_space> vals) {
    return vals;
}

template<class Type>
ILU<Type, Cuda> ilu0_impl(SparseMatrix<double, Cuda> matrix) {
    assert(matrix.nrows() == matrix.ncols());

    const size_t n = matrix.nrows();
    const size_t nnz = matrix.nnz();

    Kokkos::View<int*, typename Cuda::memory_space> row_map = matrix.row_map_;
    Kokkos::View<int*, typename Cuda::memory_space> inds = matrix.inds_;
    Kokkos::View<double*, typename Cuda::memory_space> vals ("vals", nnz);
    Kokkos::deep_copy(typename Cuda::execution_space(), vals, matrix.vals_);

    cusparseMatDescr_t descr_M = 0;
    csrilu02Info_t info_M  = 0;

    int pBufferSize;
    void *pBuffer = 0;

    CudaLibSingleton& handles = CudaLibSingleton::singleton();

    // Create Descriptors and info objects
    cusparseCreateMatDescr(&descr_M);
    cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);

    cusparseCreateCsrilu02Info(&info_M);

    // Allocate buffer
    cusparseDcsrilu02_bufferSize(handles.cusparse_handle, n, nnz,
        descr_M, vals.data(), row_map.data(), inds.data(), info_M, &pBufferSize);
    cudaMalloc((void**)&pBuffer, pBufferSize);


    // Do analysis step
    double alpha = 0;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<typename Cuda::execution_space>(0, n),
                            KOKKOS_LAMBDA(const int& i, double& update) {
        int k = row_map(i);
        const int end = row_map(i+1);
        double sum = 0;
        for (; k < end; k++) {
            sum += std::fabs(vals(k));
        }
        if (update < sum) {
            update = sum;
        }
    }, Kokkos::Max<double>(alpha));
    alpha *= std::numeric_limits<Type>::epsilon();
    cusparseDcsrilu02_numericBoost(handles.cusparse_handle, info_M, 1, &alpha, &alpha);

    cusparseDcsrilu02_analysis(handles.cusparse_handle, n, nnz, descr_M,
        vals.data(), row_map.data(), inds.data(), info_M,
        CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);

    // Do numerical step
    cusparseDcsrilu02(handles.cusparse_handle, n, nnz, descr_M,
        vals.data(), row_map.data(), inds.data(), info_M, CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer);

    // free unneeded resources
    cudaFree(pBuffer);
    cusparseDestroyMatDescr(descr_M);
    cusparseDestroyCsrilu02Info(info_M);

    Kokkos::View<Type*, typename Cuda::memory_space> vals_type = type_convert<Type>(vals);

    // Create ILU object
    return ILU<Type, Cuda>(n, nnz, row_map, inds, vals_type);
}

template<>
ILU<float, Cuda> ilu0<float, Cuda>(SparseMatrix<double, Cuda> matrix) {
    return ilu0_impl<float>(matrix);
}

template<>
ILU<double, Cuda> ilu0<double, Cuda>(SparseMatrix<double, Cuda> matrix) {
    return ilu0_impl<double>(matrix);
}
