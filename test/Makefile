CXX      = nvcc
CXXFLAGS = -g -O3 -std=c++17 --default-stream per-thread -lcublas
CXXFLAGS += --compiler-options -Wall -Wno-deprecated-gpu-targets --expt-extended-lambda -lineinfo
INCL     = -I${CUDA_HOME}/include -I${CUDA_HOME}/samples/common/inc -I${CUDA_HOME}/NVIDIA_CUDA-11.0_Samples/common/inc

SRCS=$(wildcard *.cu)
OBJS=$(SRCS:.cu=)
all: $(OBJS)

$(OBJS): %: %.cu
	$(CXX) $(INCL) $(CXXFLAGS) $< -o $@

clean:
	rm -f *.o *.lst $(OBJS) *.ptx *.cub
