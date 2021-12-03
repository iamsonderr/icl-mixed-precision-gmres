This code is designed to experiment with the use of mixed precision inside the
GMRES method for solving sparse systems of linear equations.  The main approach
is similar to mixed-precision iterative refinement.


The work was first published at the 2020 Smoky Mountains Computational Sciences
& Engineering Conference and is available online.
https://www.icl.utk.edu/publications/improving-performance-gmres-method-using-mixed-precision-techniques


Compiling the code requires Kokkos, CUDA 10, and Intel MKL.  The code is know
if compile with Kokkos 3.1.01, but likely works with other versions too.
