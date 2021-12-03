
#ifndef LOADMATRIX_HPP
#define LOADMATRIX_HPP

#include <cstdio>
#include <cstdlib>
#include <utility>

extern "C" {
	#include "mmio.h"
}

#include "types_mkl.hpp"
#include <Kokkos_Core.hpp>
#include "kernels.hpp"

template<class Scalar>
SparseMatrix<Scalar, MKL> LoadMatrix(char *file) {

	static_assert(std::is_same<MKL::memory_space, Kokkos::HostSpace>::value, "MKL mem space differs from Host space");

	FILE *in = fopen(file, "r");
	if (in == nullptr) {
		throw std::invalid_argument("Could not access file");
	}

	MM_typecode matcode;
	int err;
	if ((err = mm_read_banner(in, &matcode)) != 0) {
		fclose(in);
		if (err == MM_PREMATURE_EOF) {
			throw std::invalid_argument("Missing values in banner");
		} else if (err == MM_NO_HEADER) {
			throw std::invalid_argument("Banner is missing");
		} else if (err == MM_UNSUPPORTED_TYPE) {
			throw std::invalid_argument("Unrecognized description");
		} else {
			throw std::invalid_argument("Malformed banner with unknown error code");
		}
	}


	int M, N, nnz;
	if (mm_read_mtx_crd_size(in, &M, &N, &nnz) !=0) {
		fclose(in);
		throw std::invalid_argument("Malformed matrix size information");
	}

	if (!(mm_is_coordinate(matcode)
			&& (mm_is_real(matcode) || mm_is_integer(matcode))
			&& (mm_is_general(matcode) || mm_is_symmetric(matcode)))) {
		fclose(in);
		throw std::invalid_argument("Unsupported matrix type");
	}

	int symmetry = mm_is_general(matcode) ? 0 : 1;

	int* I = new int[nnz];
	int* J = new int[nnz];
	Scalar* V = new Scalar[nnz];

	Kokkos::View<int*, Kokkos::HostSpace> rows ("sparse matrix rows", N+1);
	Kokkos::deep_copy(rows, 1);
	rows(0) = 0;

	int true_nnz = N;

	for (int i = 0; i < nnz; i++) {
		double elt;
		fscanf(in, "%d %d %lg\n", &I[i], &J[i], &elt);
		V[i] = elt;
		I[i]--;  /* adjust from 1-based to 0-based */
		J[i]--;

		if (I[i] != J[i]) {
			// Diagonals already included
			true_nnz++;
			rows(I[i]+1) += 1; // count how many elts are in previous row
			if (symmetry) {
				true_nnz++;
				rows(J[i]+1) += 1;
			}
		}
	}

	fclose(in);

	Kokkos::View<Scalar*, Kokkos::HostSpace> vals ("sparse matrix vals", true_nnz);
	Kokkos::View<int*, Kokkos::HostSpace> inds ("sparse matrix inds", true_nnz);
	Kokkos::deep_copy(inds, -1);

	Kokkos::View<int*, Kokkos::HostSpace> row_count ("workspace - row counts", N);

	for (int i = 0; i < N; i++) {
		// turn offset from previous row into offset from start
		rows(i+1) += rows(i);
		row_count(i) = 1;
		// Add a base diagonal entry
		inds(rows(i)) = i;
		vals(rows(i)) = 0;
	}


	for (int i = 0; i < nnz; i++) {
		int row = I[i];
		int rowStart = rows(row);
		int ind = J[i];
		Scalar val = V[i];

		if (ind == row) {
			vals(rowStart) = val;
		} else {
			unsigned int j = row_count(row);
			row_count(row) += 1;
			inds(rowStart+j) = ind;
			vals(rowStart+j) = val;

			if (symmetry) {
				unsigned int j = row_count(ind);
				int rowStart = rows(ind);
				row_count(ind) += 1;
				inds(rowStart+j) = row;
				vals(rowStart+j) = val;
			}
		}
	}

	for (int i = 0; i < N; ++i) {
		int rowStart = rows(i);
		int rowEnd = rows(i+1);
		int rowLen = rowEnd - rowStart;
		for (size_t k = 0; k < rowLen-1; ++k) {
			bool no_swaps = true;
			for (int j = rowStart; j < rowEnd-1; ++j) {
				if (inds(j) > inds(j+1)) {
					std::swap(inds(j), inds(j+1));
					std::swap(vals(j), vals(j+1));
					no_swaps = false;
				}
			}
			if (no_swaps) {
				break;
			}
		}
	}

	delete [] I;
	delete [] J;
	delete [] V;

	SparseMatrix<Scalar, MKL> A (M, N, rows, inds, vals);

	return A;
}

template<class Scalar>
Vect<Scalar, MKL> LoadVector(char *file, int col=0) {
	FILE *in = fopen(file, "r");
	if (in == nullptr) {
		throw std::invalid_argument("Could not access file");
	}

	MM_typecode matcode;
	int err;
	if ((err = mm_read_banner(in, &matcode)) != 0) {
		fclose(in);
		if (err == MM_PREMATURE_EOF) {
			throw std::invalid_argument("Missing values in banner");
		} else if (err == MM_NO_HEADER) {
			throw std::invalid_argument("Banner is missing");
		} else if (err == MM_UNSUPPORTED_TYPE) {
			throw std::invalid_argument("Unrecognized description");
		} else {
			throw std::invalid_argument("Malformed banner with unknown error code");
		}
	}


	int M, N, nnz;
	if (mm_is_array(matcode)) {
		err = mm_read_mtx_array_size(in, &M, &N);
	} else {
		err = mm_read_mtx_crd_size(in, &M, &N, &nnz);
	}
	if (err !=0) {
		fclose(in);
		throw std::invalid_argument("Malformed matrix size information");
	}

	if (col >= N) {
		fclose(in);
		std::stringstream errMsg;
		errMsg << "Column " << col << " is too large for the " << N << " vectors";
		throw std::invalid_argument(errMsg.str().c_str());
	}

	Vect<Scalar, MKL> result(M);
	Scalar* result_data = result.data();

	if (mm_is_array(matcode)){

		// run past the first col-1 columns
		double elt;
		for (int i = 0; i < col; i++) {
			for (int j = 0; j < M; j++) {
				fscanf(in, "%lf\n", &elt);
			}
		}
		for (int j = 0; j < M; j++) {
			fscanf(in, "%lf\n", &elt);
			result_data[j] = Scalar(elt);
		}
	} else if (mm_is_coordinate(matcode)) {
		fill(0.0, result);

		for (int i = 0; i < nnz; i++) {
			int ii, jj;
			double vv;
			fscanf(in, "%d %d %lg\n", &ii, &jj, &vv);
			--ii;
			--jj;
			if (jj == col) {
				result_data[ii] = Scalar(vv);
			}
		}
	} else {
		std::cerr << "Unknown matrix type" << std::endl;
		exit(-1);
	}

	fclose(in);
	return result;
}

#endif // LOADMATRIX_HPP
