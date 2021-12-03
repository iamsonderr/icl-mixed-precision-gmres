#ifndef TYPES_MKL_HPP
#define TYPES_MKL_HPP

#include<cstdlib>

#include<mkl_spblas.h>

#include "types.hpp"


struct MKL {
public:
    typedef Kokkos::HostSpace memory_space;
    typedef Kokkos::OpenMP execution_space;
};

template<class Type>
class SparseMatrix<Type, MKL> {
private:


    int m_, n_;

    sparse_matrix_t mkl_handle_;
    struct matrix_descr mkl_desc_;

    void create_mkl_handles();

    template<class, class>
    friend class SparseMatrix;

public:

    Kokkos::View<int*, typename MKL::memory_space> row_map_;
    Kokkos::View<int*, typename MKL::memory_space> inds_;
    Kokkos::View<Type*,   typename MKL::memory_space> vals_;

    SparseMatrix(int m, int n,
                 Kokkos::View<int*, typename MKL::memory_space> row_map,
                 Kokkos::View<int*, typename MKL::memory_space> inds,
                 Kokkos::View<Type*, typename MKL::memory_space> vals)
         : m_(m), n_(n), row_map_(row_map), inds_(inds), vals_(vals) {
            create_mkl_handles();
    }

    template<class OldType>
    SparseMatrix(SparseMatrix<OldType, MKL> old)
            : m_(old.m_), n_(old.n_),
              row_map_(old.row_map_), inds_(old.inds_),
              vals_("sparse matrix vals", old.vals_.extent(0)) {
        copy_vals(old.vals_);
        create_mkl_handles();
    }

    // Kokkos lambda's don't work in constructors
    template<class OldType>
    void copy_vals(Kokkos::View<OldType*, typename MKL::memory_space> old_vals) {
        Kokkos::parallel_for(Kokkos::RangePolicy<typename MKL::execution_space>(0, vals_.extent(0)),
                             KOKKOS_LAMBDA(const size_t& i) {
            vals_(i) = old_vals(i);
        });
    }

    template<class OldDevice>
    SparseMatrix(SparseMatrix<Type, OldDevice> old)
            : m_(old.m_), n_(old.n),
              row_map_("sparse matrix rows", old.row_map_.extent(0)),
              inds_("sparse matrix inds", old.inds_.extent(0)),
              vals_("sparse matrix vals", old.vals_.extent(0)) {
        Kokkos::deep_copy(row_map_, old.row_map_);
        Kokkos::deep_copy(inds_, old.inds_);
        Kokkos::deep_copy(vals_, old.vals_);
        create_mkl_handles();
    }

    int nrows() const {
        return m_;
    }
    int ncols() const {
        return n_;
    }

    int row_map(int row) const {
        return row_map_(row);
    }
    int inds(int offset) const {
        return inds_(offset);
    }
    Type vals(int offset) const {
        return vals_(offset);
    }

    sparse_matrix_t mkl_handle() const {
        return mkl_handle_;
    }
    struct matrix_descr mkl_desc() const {
        return mkl_desc_;
    }

    Vect<Type, MKL> vals_vect() {
        return Vect<Type, MKL>(vals_);
    }

    bool is_transposed() {
        return false;
    }
};


template<class Type>
class ILU<Type, MKL>: public LinearOperator<Type, MKL> {
private:

    int n_;

    sparse_matrix_t mkl_handle_;
    struct matrix_descr mkl_L_desc_;
    struct matrix_descr mkl_U_desc_;

    void create_mkl_handles();

    template<class, class>
    friend class ILU;

    template<class, class>
    friend class ILU_Jacobi;

public:
    Kokkos::View<int*, typename MKL::memory_space> row_map_;
    Kokkos::View<int*, typename MKL::memory_space> inds_;
    Kokkos::View<Type*,   typename MKL::memory_space> vals_;

    ILU(int n,
        Kokkos::View<int*, typename MKL::memory_space> row_map,
        Kokkos::View<int*, typename MKL::memory_space> inds,
        Kokkos::View<Type*, typename MKL::memory_space> vals)
         : n_(n), row_map_(row_map), inds_(inds), vals_(vals) {
             create_mkl_handles();
     }

     // template<class OldType>
     // ILU(ILU<OldType, MKL> old)
     //        : n_(old.n_),
     //          row_map_(old.row_map_), inds_(old.inds_),
     //          vals_("vals", old.vals_.extent(0)){
     //    Kokkos::deep_copy(vals_, old.vals_);
     //    create_mkl_handles();
     // }

    int n() const {
        return n_;
    }

    int nnz() const {
        return row_map_(n_);
    }

    int row_map(int row) const {
        return row_map_(row);
    }
    int inds(int offset) const {
        return inds_(offset);
    }
    Type vals(int offset) const {
        return vals_(offset);
    }

    int* row_map_data() const {
        return row_map_.data();
    }

    int* inds_data() const {
        return inds_.data();
    }

    Type* vals_data() const {
        return vals_.data();
    }

    sparse_matrix_t mkl_handle() const {
        return mkl_handle_;
    }
    struct matrix_descr mkl_L_desc() const {
        return mkl_L_desc_;
    }
    struct matrix_descr mkl_U_desc() const {
        return mkl_U_desc_;
    }


    void apply(Vect<Type, MKL> rhs) {
        ilusv(*this, rhs);
    }
};


template<>
class ILU_Jacobi_handles<MKL> {
    struct matrix_descr L_desc_;
    struct matrix_descr U_desc_;

    sparse_matrix_t matrix_;

    void create_matrix_handle(ILU_Jacobi<double, MKL>& ilu) {
        assert(mkl_sparse_d_create_csr(&matrix_, SPARSE_INDEX_BASE_ZERO,
                                       ilu.n(), ilu.n(),
                                       ilu.row_map_data(), ilu.row_map_data()+1,
                                       ilu.inds_data(), ilu.vals_data())
               == SPARSE_STATUS_SUCCESS);
    }
    void create_matrix_handle(ILU_Jacobi<float, MKL>& ilu) {
        assert(mkl_sparse_s_create_csr(&matrix_, SPARSE_INDEX_BASE_ZERO,
                                       ilu.n(), ilu.n(),
                                       ilu.row_map_data(), ilu.row_map_data()+1,
                                       ilu.inds_data(), ilu.vals_data())
               == SPARSE_STATUS_SUCCESS);
    }

public:

    template<class Type>
    ILU_Jacobi_handles(ILU_Jacobi<Type, MKL>& ilu) {
        L_desc_.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
        L_desc_.mode = SPARSE_FILL_MODE_LOWER;
        L_desc_.diag = SPARSE_DIAG_UNIT;

        U_desc_.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
        U_desc_.mode = SPARSE_FILL_MODE_UPPER;
        U_desc_.diag = SPARSE_DIAG_NON_UNIT;

    }

    struct matrix_descr L_desc() {
        return L_desc_;
    }

    struct matrix_descr U_desc() {
        return U_desc_;
    }

    sparse_matrix_t matrix() {
        return matrix_;
    }
};

#endif // TYPES_MKL_HPP
