#ifndef TYPES_CUDA_HPP
#define TYPES_CUDA_HPP

#include "kernels.hpp"

#include<cublas_v2.h>
#include<cusparse.h>

struct CudaLibSingleton {
    cublasHandle_t cublas_handle;
    cusparseHandle_t cusparse_handle;

    CudaLibSingleton() {
        auto cublas_status = cublasCreate(&cublas_handle);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            Kokkos::abort("CUBLAS initialization failed\n");
        }

        auto cusparse_status = cusparseCreate(&cusparse_handle);
        if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
            cublasDestroy(cublas_handle);

            Kokkos::abort("CUSPARSE initialization failed\n");
        }

        Kokkos::push_finalize_hook ([&] () {
            cublasDestroy(cublas_handle);
            cusparseDestroy(cusparse_handle);
        });
     }

  static CudaLibSingleton & singleton() {
      static CudaLibSingleton s;
      return s;
  }
};


struct Cuda {
public:
    typedef Kokkos::CudaSpace memory_space;
    //typedef Kokkos::CudaUVMSpace memory_space;
    typedef Kokkos::Cuda execution_space;
};


template<class Type>
class SparseMatrix<Type, Cuda>{
private:

    std::shared_ptr<cusparseMatDescr_t> desc_;

    void create_cuda_handles() {
        desc_ = std::shared_ptr<cusparseMatDescr_t>(new cusparseMatDescr_t(),
                                                    [] (cusparseMatDescr_t* desc) {
                                                        cusparseDestroyMatDescr(*desc);
                                                        delete desc;
                                                    });
        cusparseCreateMatDescr(desc_.get());
    }

    bool transposed_ = false;

public:

    int m_, n_, nnz_;

    Kokkos::View<int*, typename Cuda::memory_space> row_map_;
    Kokkos::View<int*, typename Cuda::memory_space> inds_;
    Kokkos::View<Type*,   typename Cuda::memory_space> vals_;


    SparseMatrix<Type, Cuda>(int m, int n,
                             Kokkos::View<int*, typename Cuda::memory_space> row_map,
                             Kokkos::View<int*, typename Cuda::memory_space> inds,
                             Kokkos::View<Type*, typename Cuda::memory_space> vals)
             : m_(m), n_(n), nnz_(inds.extent(0)),
             row_map_(row_map), inds_(inds), vals_(vals) {
        create_cuda_handles();
     }

    template<class OldType>
    SparseMatrix(SparseMatrix<OldType, Cuda> old)
             : m_(old.m_), n_(old.n_), nnz_(old.inds_.extent(0)),
               row_map_(old.row_map_),
               inds_(old.inds_),
               vals_("vals", old.vals_.extent(0)),
               transposed_(old.is_transposed()) {
        copy_vals(old.vals_);
        create_cuda_handles();
    }

    // Kokkos lambda's don't work in constructors
    template<class OldType>
    void copy_vals(Kokkos::View<OldType*, typename Cuda::memory_space> old_vals) {
        auto vals = vals_;
        Kokkos::parallel_for(Kokkos::RangePolicy<typename Cuda::execution_space>(0, vals_.extent(0)),
                             KOKKOS_LAMBDA(const size_t& i) {
            vals(i) = old_vals(i);
        });
    }

    template<class OldDevice>
    SparseMatrix(SparseMatrix<Type, OldDevice> old)
            : m_(old.m_), n_(old.n_), nnz_(old.inds_.extent(0)),
              row_map_("row_map", old.row_map_.extent(0)),
              inds_("inds", old.inds_.extent(0)),
              vals_("vals", old.vals_.extent(0)),
              transposed_(old.is_transposed()) {
        Kokkos::deep_copy(row_map_, old.row_map_);
        Kokkos::deep_copy(inds_, old.inds_);
        Kokkos::deep_copy(vals_, old.vals_);
        create_cuda_handles();
    }

    int nrows() const {
        return m_;
    }
    int ncols() const {
        return n_;
    }

    int nnz() const {
        return nnz_;
    }

    int* row_map_data() {
        return row_map_.data();
    }
    int* inds_data() {
        return inds_.data();
    }
    Type* vals_data() {
        return vals_.data();
    }

    Vect<Type, Cuda> vals_vect() {
        return Vect<Type, Cuda>(vals_);
    }

    cusparseMatDescr_t desc() {
        return *desc_;
    }

    void set_transpose(bool new_trans) {
        this->transposed_ = new_trans;
    }

    bool is_transposed() {
        return this->transposed_;
    }
};

class ILU_handles {
public:
    cusparseMatDescr_t cuda_L_desc_;
    csrsv2Info_t cuda_L_info_;
    cusparseMatDescr_t cuda_U_desc_;
    csrsv2Info_t cuda_U_info_;

    ILU_handles(cusparseMatDescr_t cuda_L_desc,
                csrsv2Info_t cuda_L_info,
                cusparseMatDescr_t cuda_U_desc,
                csrsv2Info_t cuda_U_info)
        : cuda_L_desc_(cuda_L_desc), cuda_L_info_(cuda_L_info),
          cuda_U_desc_(cuda_U_desc), cuda_U_info_(cuda_U_info) {
    }

    ~ILU_handles() {
        cusparseDestroyMatDescr(cuda_L_desc_);
        cusparseDestroyCsrsv2Info(cuda_L_info_);
        cusparseDestroyMatDescr(cuda_U_desc_);
        cusparseDestroyCsrsv2Info(cuda_U_info_);
    }
};

template<class Type>
class ILU<Type, Cuda>: public LinearOperator<Type, Cuda> {
private:

    int n_, nnz_;

    Kokkos::View<int*, typename Cuda::memory_space> row_map_;
    Kokkos::View<int*, typename Cuda::memory_space> inds_;
    Kokkos::View<Type*,   typename Cuda::memory_space> vals_;

    std::shared_ptr<ILU_handles> handles_;

    void create_cuda_handles();

    template<class, class>
    friend class ILU;

    template<class, class>
    friend class ILU_Jacobi;

public:
    ILU<Type, Cuda>(int n,
                    int nnz,
                    Kokkos::View<int*, typename Cuda::memory_space> row_map,
                    Kokkos::View<int*, typename Cuda::memory_space> inds,
                    Kokkos::View<Type*, typename Cuda::memory_space> vals)
             : n_(n), nnz_(nnz),
              row_map_(row_map), inds_(inds), vals_(vals) {
        create_cuda_handles();
    }

    int n() const {
        return n_;
    }

    int nnz() const {
        return nnz_;
    }

    int* row_map_data() {
        return row_map_.data();
    }
    int* inds_data() {
        return inds_.data();
    }
    Type* vals_data() {
        return vals_.data();
    }

    cusparseMatDescr_t cuda_L_desc() const {
        return handles_->cuda_L_desc_;
    }
    csrsv2Info_t cuda_L_info() const {
        return handles_->cuda_L_info_;
    }
    cusparseMatDescr_t cuda_U_desc() const {
        return handles_->cuda_U_desc_;
    }
    csrsv2Info_t cuda_U_info() const {
        return handles_->cuda_U_info_;
    }

    void apply(Vect<Type, Cuda> rhs) {
        ilusv(*this, rhs);
    }
};

template<>
class ILU_Jacobi_handles<Cuda> {
public:

    template<class Type>
    ILU_Jacobi_handles(const ILU_Jacobi<Type, Cuda>&) {
    }
    ~ILU_Jacobi_handles<Cuda>() {
    }
};

#endif // TYPES_CUDA_HPP
