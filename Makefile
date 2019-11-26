CXX      = nvcc
#CXXFLAGS = -g -O3 -std=c++11 --compiler-options -Wall -arch=sm_60 -Wno-deprecated-gpu-targets -lcublas # Pascal
CXXFLAGS = -g -O3 -std=c++11 --compiler-options -Wall -arch=sm_70 -Wno-deprecated-gpu-targets -lcublas # Volta
INCL     = -I${CUDA_HOME}/include -I${CUDA_HOME}/samples/common/inc

SRCS=$(wildcard *.cu)
OBJS=$(SRCS:.cu=)
all: $(OBJS)

$(OBJS): %: %.cu
	$(CXX) $(INCL) $(CXXFLAGS) $< -o $@

clean:
	rm -f *.o *.lst $(OBJS) *.ptx *.cub
