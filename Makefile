CXX      = nvcc
CXXFLAGS = -g -O3 -std=c++11 --compiler-options -Wall -arch=sm_70 -Wno-deprecated-gpu-targets -lcublas
SRCS     = $(shell ls *.cu)
OBJS     = $(SRCS:.cu=.o)
EXEC     = simpleGEMM
INCL     = -I${CUDA_HOME}/include

all: miniapp

miniapp: $(OBJS)
	$(CXX) $(INCL) $(CXXFLAGS) $(OBJS) -o $(EXEC)

%.o: %.cu
	$(CXX) -c $(INCL) $(CXXFLAGS) $< -o $@

clean:
	rm -f *.o *.lst $(EXEC) *.ptx *.cub
