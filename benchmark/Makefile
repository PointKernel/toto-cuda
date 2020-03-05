CXX      = nvcc
CXXFLAGS = -g -O3 -std=c++11 --default-stream per-thread -lcublas
CXXFLAGS += --compiler-options -Wall -Wno-deprecated-gpu-targets
INCL     = -I${CUDA_HOME}/include -I${CUDA_HOME}/samples/common/inc

SRCS=$(wildcard *.cu)
OBJS=$(SRCS:.cu=)
all: $(OBJS)

$(OBJS): %: %.cu
	$(CXX) $(INCL) $(CXXFLAGS) $< -o $@

clean:
	rm -f *.o *.lst $(OBJS) *.ptx *.cub
