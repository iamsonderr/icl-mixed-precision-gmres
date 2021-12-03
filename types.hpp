
#ifndef TYPES_HPP
#define TYPES_HPP

#include<cassert>
#include<memory>

#include<Kokkos_Core.hpp>

template<class Type, class Device>
class Vect;
template<class Type, class Device>
class MultiVect;

template<class Type, class Device>
class Scalar {
private:
    typedef Kokkos::View<Type, Kokkos::LayoutLeft, typename Device::memory_space> view_type;
    view_type view_;

public:
    Scalar()
        : view_ ("Scalar") {
    }

    Scalar(Type val)
        : view_ ("Scalar") {
        view_() = val;
    }

    Scalar(Vect<Type, Device> vec, size_t idx)
        : view_(vec.view(), idx) {
    }

    Scalar(MultiVect<Type, Device> vec, size_t row, size_t col)
        : view_(vec.view(), row, col) {
    }

    Type access() {
        if (Kokkos::SpaceAccessibility<Kokkos::OpenMP, typename Device::memory_space>::accessible) {
            return view_();
        } else {
            auto temp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view_);
            return temp();
        }
    }

    Type* data() {
        return view_.data();
    }

    view_type view() {
        return view_;
    }
};

template<class Type, class Device>
class Vect {
private:
    typedef Kokkos::View<Type*, Kokkos::LayoutLeft, typename Device::memory_space> view_type;
    view_type view_;

public:
    Vect() {
    }
    Vect(size_t n)
        : view_("Vect::view", n) {
    }
    Vect(view_type view)
        : view_(view) {
        }

    Vect(Vect<Type, Device> vec, Kokkos::pair<size_t, size_t> rows)
        : view_(vec.view(), rows) {
    }

    Vect(MultiVect<Type, Device> vec, decltype(Kokkos::ALL), size_t col)
        : view_(vec.view(), Kokkos::ALL, col) {
    }
    Vect(MultiVect<Type, Device> vec, Kokkos::pair<size_t, size_t> rows, size_t col)
        : view_(vec.view(), rows, col) {
    }

    Scalar<Type, Device> operator()(const size_t i) {
        return Scalar<Type, Device>(*this, i);
    }

    Vect<Type, Device> operator()(Kokkos::pair<size_t, size_t> rows) {
        return Vect<Type, Device>(*this, rows);
    }

    Type* data() {
        return view_.data();
    }

    size_t n() const {
        return view_.extent(0);
    }

    view_type view() {
        return view_;
    }

    Type access(const size_t i) {
        if (Kokkos::SpaceAccessibility<Kokkos::OpenMP, typename Device::memory_space>::accessible) {
            return view_(i);
        } else {
            Kokkos::View<Type, Kokkos::LayoutLeft, typename Device::memory_space> view_value(view_, i);
            auto temp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view_value);
            return temp();
        }
    }
};

template<class Type, class Device>
class MultiVect {
private:
    typedef Kokkos::View<Type**, Kokkos::LayoutLeft, typename Device::memory_space> view_type;
    view_type view_;

    bool transposed_ = false;

    MultiVect(view_type view, bool transposed)
        : view_(view), transposed_(transposed) {
    }

public:
    MultiVect() {
    }

    MultiVect(size_t m, size_t n)
        : view_("MultiVect::view", m, n) {
    }

    MultiVect(MultiVect<Type, Device> vec, Kokkos::pair<size_t, size_t> rows, decltype(Kokkos::ALL))
            : transposed_(vec.transposed_) {
        if (transposed_) {
            view_ = view_type(vec.view(), Kokkos::ALL, rows);
        } else {
            view_ = view_type(vec.view(), rows, Kokkos::ALL);
        }

        assert(nrows() == rows.second - rows.first);
        assert(ncols() == vec.ncols());
    }
    MultiVect(MultiVect<Type, Device> vec, decltype(Kokkos::ALL), Kokkos::pair<size_t, size_t> cols)
            : transposed_(vec.transposed_) {
        if (transposed_) {
            view_ = view_type(vec.view(), cols, Kokkos::ALL);
        } else {
            view_ = view_type(vec.view(), Kokkos::ALL, cols);
        }

        assert(ncols() == cols.second - cols.first);
        assert(nrows() == vec.nrows());
    }
    MultiVect(MultiVect<Type, Device> vec, Kokkos::pair<size_t, size_t> rows, Kokkos::pair<size_t, size_t> cols)
            : transposed_(vec.transposed_) {
        if (transposed_) {
            view_ = view_type(vec.view(), cols, rows);
        } else {
            view_ = view_type(vec.view(), rows, cols);
        }

        assert(nrows() == rows.second - rows.first);
        assert(ncols() == cols.second - cols.first);
    }

    size_t nrows() {
        if (transposed_) {
            return view_.extent(1);
        } else {
            return view_.extent(0);
        }
    }

    size_t ncols() {
        if (transposed_) {
            return view_.extent(0);
        } else {
            return view_.extent(1);
        }
    }

    size_t nrows_base() {
        return view_.extent(0);
    }

    size_t ncols_base() {
        return view_.extent(1);
    }

    Type* data() {
        return view_.data();
    }

    size_t stride() const {
        return view_.stride(1);
    }

    size_t n() const {
        return view_.extent(0);
    }

    bool transposed() const {
        return transposed_;
    }

    MultiVect<Type, Device> transpose_matrix() {
        return MultiVect<Type, Device>(view_, !transposed_);
    }

    Scalar<Type, Device> operator()(size_t i, size_t j) {
        if (transposed_) {
            return Scalar<Type, Device>(*this, j, i);
        } else {
            return Scalar<Type, Device>(*this, i, j);
        }
    }

    Vect<Type, Device> operator()(Kokkos::pair<size_t, size_t> rows, size_t col) {
        return Vect<Type, Device>(*this, rows, col);
    }

    view_type view() {
        return view_;
    }
};

template<class Type, class Device>
class LinearOperator {
public:
    virtual ~LinearOperator() {
    };

    virtual void apply(Vect<Type, Device> rhs) = 0;
};

template<class Type, class Device>
class SparseMatrix {
};

template<class Type, class Device>
class ILU: public LinearOperator<Type, Device> {
};

template<class Device>
class ILU_Jacobi_handles {
};

template<class Type, class Device>
class ILU_Jacobi: public LinearOperator<Type, Device> {
private:

    const int n_, nnz_, steps_;

    Kokkos::View<int*,  typename Device::memory_space> row_map_;
    Kokkos::View<int*,  typename Device::memory_space> inds_;
    Kokkos::View<Type*, typename Device::memory_space> vals_;
    Vect<Type, Device> diag_;
    Kokkos::View<int*,  typename Device::memory_space> diag_inds_;

    Vect<Type, Device> temp1_;
    Vect<Type, Device> temp2_;

    std::shared_ptr<ILU_Jacobi_handles<Device>> handles_;


public:
    ILU_Jacobi<Type, Device>(
                    int n,
                    int nnz,
                    int steps,
                    Kokkos::View<int*, typename Device::memory_space> row_map,
                    Kokkos::View<int*, typename Device::memory_space> inds,
                    Kokkos::View<Type*, typename Device::memory_space> vals)
             : n_(n), nnz_(nnz), steps_(steps),
              row_map_(row_map), inds_(inds), vals_(vals),
              diag_(n), diag_inds_("diag inds", n),
              temp1_(n), temp2_(n) {
        create_handles();
    }

    ILU_Jacobi<Type, Device>(ILU<Type, Device> ilu, int steps)
            : n_(ilu.n()), nnz_(ilu.nnz()), steps_(steps),
              row_map_(ilu.row_map_), inds_(ilu.inds_), vals_(ilu.vals_),
              diag_(n_), diag_inds_("diag inds", n_),
              temp1_(n_), temp2_(n_) {
        create_handles();
    }

    void create_handles() {
        handles_ = std::make_shared<ILU_Jacobi_handles<Device>>(*this);

        const auto row_map = row_map_;
        const auto inds    = inds_;
        const auto vals    = vals_;
              auto diag    = diag_.view();
              auto diag_inds = diag_inds_;
        Kokkos::parallel_for(Kokkos::RangePolicy<typename Device::execution_space>(0, n_),
                             KOKKOS_LAMBDA(const size_t& i) {
            auto j = row_map(i);
            while (inds(j) < i) {
                ++j;
            }
            diag(i) = 1/vals(j);
            diag_inds(i) = j;
        });
    }

    int n() const {
        return n_;
    }

    int nnz() const {
        return nnz_;
    }

    int steps() const {
        return steps_;
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

    Type* diag_inds_data() {
        return diag_inds_.data();
    }

    Type* diag_data() {
        return diag_.data();
    }

    Vect<Type, Device> diag_vect() {
        return diag_;
    }

    Kokkos::View<int*,  typename Device::memory_space> row_map_view() {
        return row_map_;
    }
    Kokkos::View<int*,  typename Device::memory_space> inds_view() {
        return inds_;
    }
    Kokkos::View<Type*, typename Device::memory_space> vals_view() {
        return vals_;
    }
    Kokkos::View<int*,  typename Device::memory_space> diag_inds_view() {
        return diag_inds_;
    }

    Vect<Type, Device>& temp1() {
        return temp1_;
    }
    Vect<Type, Device>& temp2() {
        return temp2_;
    }

    ILU_Jacobi_handles<Device>& handles() const {
        return *handles_;
    }

    void apply(Vect<Type, Device> rhs) {
        ilusv_jacobi(*this, rhs);
    }
};

template<class Type, class Device>
class Identity: public LinearOperator<Type, Device> {
    void apply(Vect<Type, Device> rhs) {
    }
};


template<class Type, class Device>
class Jacobi: public LinearOperator<Type, Device> {
private:

    const int n_;
    Vect<Type, Device> diag_;

public:

    Jacobi<Type, Device>(SparseMatrix<Type, Device> A)
            : n_(A.nrows()), diag_(n_) {
        get_diag_vals(A);
    }


    // Can't create device lambdas in the constructor or in private methods
    void get_diag_vals(SparseMatrix<Type, Device> A) {
        const auto row_map = A.row_map_;
        const auto inds    = A.inds_;
        const auto vals    = A.vals_;
              auto diag    = diag_.view();

        // epsilon*norm(A)
        Type alpha = 0;
        Kokkos::parallel_reduce(Kokkos::RangePolicy<typename Device::execution_space>(0, n_),
                                KOKKOS_LAMBDA(const int& i, Type& update) {
            int k = row_map(i);
            const int end = row_map(i+1);
            Type sum = 0;
            for (; k < end; k++) {
                sum += std::fabs(vals(k));
            }
            if (update < sum) {
                update = sum;
            }
        }, Kokkos::Max<Type>(alpha));
        alpha *= std::numeric_limits<float>::epsilon();

        Kokkos::parallel_for(Kokkos::RangePolicy<typename Device::execution_space>(0, n_),
                             KOKKOS_LAMBDA(const size_t& i) {
            auto j = row_map(i);
            while (inds(j) < i) {
                ++j;
            }
            if (vals(j) >= 0) {
                diag(i) = 1/((vals(j) < alpha) ? alpha : vals(j));
            } else {
                diag(i) = 1/((vals(j) > -alpha) ? -alpha : vals(j));
            }
        });
    }

    int n() const {
        return n_;
    }

    Type* diag_data() {
        return diag_.data();
    }

    Vect<Type, Device> diag_vect() {
        return diag_;
    }

    void apply(Vect<Type, Device> rhs) {
        gdmv(1.0, diag_, rhs, 0.0, rhs);
    }
};

#endif // TYPES_HPP
