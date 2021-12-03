
#include <random>

#include "kernels.hpp"
#include "types_mkl.hpp"
#include "types_cuda.hpp"

#include "LoadMatrix.hpp"

Vect<double, MKL> rand_vect(int n, typename std::mt19937::result_type seed=0) {
    std::mt19937 engine(seed);
    // note floats are generated so that the values are the same regardless whether x is single or double
    std::uniform_real_distribution<float> dist;

    Vect<double, MKL> x (n);
    double* x_data = x.data();
    // Use regular loop so that the vector is deterministic for a given seed
    for (unsigned int i = 0; i<x.n(); i++) {
    x_data[i] = dist(engine);
    }
    return x;
}

template<class Type, class Device>
void rand_fill(Vect<Type, Device> x, typename std::mt19937::result_type seed=0) {
    Vect<double, MKL> generated = rand_vect(x.n(), seed);
    Kokkos::deep_copy(x.view(), generated.view());
}

int klein_lu_bound(double eps, double delta, int n) {
    double log_2n = std::log(2*n);
    return int(std::ceil((log_2n*log_2n - std::log(eps*delta*delta))/eps));
}


template<class Type, class Device>
void condest(SparseMatrix<Type, Device> A, int rand_seed, int max_iters) {

    const int n = A.nrows();

    double eps = std::numeric_limits<double>::epsilon();
    double c1 = 8*eps;
    double erfinv_c2 = 8.862271574665521045654E-4;
    double c3 = 1/(64*eps);
    double c4 = std::sqrt(eps);
    double c1_prime = 4*eps;
    int power_iter_tol = klein_lu_bound(0.1, 1e-12, n);

    SparseMatrix<Type, Device> A_trans = A;
    A_trans.set_transpose(true);

    Vect<Type, Device> v_max (n);
    rand_fill(v_max, rand_seed+5);
    double sigma_max = power_iteration(A, v_max, power_iter_tol);

    Vect<Type, Device> v_min (n);
    copy(v_max, v_min);
    double sigma_min = sigma_max;

    Vect<Type, Device> x_exact (n);
    rand_fill(x_exact, rand_seed);
    double x_rand_norm = nrm2(x_exact);
    scal(1/x_rand_norm, x_exact);

    Vect<Type, Device> b (n);
    spmv(1.0, A, x_exact, 0.0, b);
    double b_norm = nrm2(b);
    double beta = b_norm;

    Vect<Type, Device> u (n);
    scal(1/beta, b, u);

    Vect<Type, Device> v (n);
    spmv(1.0, A_trans, u, 0.0, v);
    double alpha = nrm2(v);
    scal(1/alpha, v);

    Vect<Type, Device> w (n);
    copy(v, w);

    Vect<Type, Device> x (n);
    fill(0.0, x);

    Vect<Type, Device> d (n);
    Vect<Type, Device> Ad (n);
    double d_norm, Ad_norm;

    double phi_bar = beta, rho_bar = alpha;
    double phi, rho, c, s, theta;

    double tau = std::sqrt(2)*erfinv_c2/x_rand_norm;
    int T = max_iters;

    std::cout << "sigma_max = " << sigma_max << std::endl;

    int t;
    for (t = 1; t <= T; t++) {

        spmv(1.0, A, v, -alpha, u);
        beta = nrm2(u);
        scal(1/beta, u);

        spmv(1.0, A_trans, u, -beta, v);
        alpha = nrm2(v);
        scal(1/alpha, v);

        rho = std::sqrt(rho_bar*rho_bar + beta*beta);
        c = rho_bar/rho;
        s = beta/rho;
        theta = s*alpha;
        rho_bar = -c*alpha;
        phi = c*phi_bar;
        phi_bar = s*phi_bar;

        axpy(phi/rho, w, x);
        scal(-theta/rho, w);
        axpy(1.0, v, w);

        //TODO lines 29-30

        copy(x_exact, d);
        axpy(-1.0, x, d);
        d_norm = nrm2(d);
        if (d_norm == 0) {
            break;
        }

        spmv(1.0, A, d, 0.0, Ad);
        Ad_norm = nrm2(Ad);
        if (Ad_norm < sigma_min*d_norm) {
            sigma_min = Ad_norm/d_norm;
            copy(d, v_min);
        }

        if (std::isnan(Ad_norm)) {
            break;
        }

        if (sigma_min/sigma_max <= c4) {
            c1 = c1_prime;
        }

        if (T == max_iters) {
            double x_norm = nrm2(x);
            if (Ad_norm/(sigma_max*x_norm+b_norm) <= c1
                || d_norm <= tau
                || sigma_max/sigma_min >= c3) {

                T = int(std::ceil(t*1.25));

                std::cout << "t = " << t << ": finishing" << std::endl;
            }

            if (t%10000 == 0) {
                std::cout << "t = " << t << ": sigma_min = " << sigma_min << std::endl;
            }
        }
    }

    // TODO lines 42-43

    std::cout << t << " iterations total" << std::endl;
    std::cout << "Computed cond(A) = " << sigma_max/sigma_min << " = " << sigma_max << "/" << sigma_min << std::endl;

}

template<class Type, class Device>
double power_iteration(SparseMatrix<Type, Device> A, Vect<Type, Device> x, int iter_count) {
    Vect<Type, Device> y (x.n());
    Type lambda;

    for (int i = 0; i < iter_count; i++) {
        spmv(1.0, A, x, 0.0, y);
        lambda = nrm2(y);
        scal(1/lambda, y, x);
    }

    return lambda;
}






int main(int argc, char* argv[]) {

    char* APath = nullptr;
    int rand_seed = 42;
    bool use_gpu = false;
    int max_iters = 100000;

    for (int i = 1; i < argc; i++) {
        if (strcmp("--Apath", argv[i]) == 0) {
            APath = argv[++i];
        } else if (strcmp("--rand", argv[i]) == 0) {
            rand_seed = std::stoi(argv[++i]);
        } else if (strcmp("--gpu", argv[i]) == 0) {
            use_gpu = true;
        } else if (strcmp("--max-iters", argv[i]) == 0) {
            max_iters = std::stoi(argv[++i]);
        } else {
            std::cout << "Unknown flag" << argv[i] << std::endl;
            return 1;
        }
    }

    if (APath == nullptr) {
        std::cout << "No value suplied for A" << std::endl;
        return 1;
    }

    Kokkos::ScopeGuard kokkos(argc, argv);

    const SparseMatrix<double, MKL> A_host = LoadMatrix<double>(APath);
    const int n = A_host.nrows();

    if (use_gpu) {
        SparseMatrix<double, Cuda> A (A_host);
        condest<double, Cuda>(A, rand_seed, max_iters);
    } else {
        std::cout << "CPU not currently supported" << std::endl;
    }


    return 0;
}
