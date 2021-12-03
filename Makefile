KOKKOS_PATH=../kokkos-include
KOKKOS_DEVICES=OpenMP,Cuda
KOKKOS_ARCH=HSW,Volta70
KOKKOS_CUDA_OPTIONS=enable_lambda

CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
CXXFLAGS=-ggdb -O3 -I. -m64 -I${MKLROOT}/include

CC = gcc
CCFLAGS=-ggdb -G -O2 -std=c11 -Wunused-parameter -Wwrite-strings

LINK = $(CXX)
LINKFLAGS = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl -lcusparse -lcublas

HEADERS=gmres.hpp IterUtil.hpp Orthogonalization.hpp kernels.hpp types.hpp types_mkl.hpp types_cuda.hpp mmio.h LoadMatrix.hpp

OBJ=gmres.o mmio.o kernels_mkl.o kernels_cuda.o

default: gmres_perf_test
.PHONY: clean


include $(KOKKOS_PATH)/Makefile.kokkos



mmio.o: mmio.c mmio.h
	$(CC) -c $< $(GCC_FLAGS)

%.o:%.cpp $(KOKKOS_CPP_DEPENDS) $(HEADERS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $< -o $(notdir $@)

PERF_TEST_OBJ = gmres_perf_test.o $(OBJ)
gmres_perf_test: $(PERF_TEST_OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(PERF_TEST_OBJ) $(KOKKOS_LIBS) $(LIB) -o gmres_perf_test

#kernel_perf_test: kernel_perf_test.cpp LATypes.hpp LoadMatrix.hpp mmio.o
#	icpc -o kernel_perf_test kernel_perf_test.cpp mmio.o $(GPP_FLAGS) $(LINKER_FLAGS) $(INCLUDE_FLAGS)

condest: condest.o $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) condest.o $(OBJ) $(KOKKOS_LIBS) $(LIB) -o condest

clean:
	-rm gmres_perf_test *.o
