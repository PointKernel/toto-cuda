CXX      = clang++
CXXFLAGS += -g -O3 -std=c++14 -W -Wall -fopenmp 
CXXFLAGS += -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${CUDA_HOME}
#INCL     = -I${CUDA_HOME}/include -I${CUDA_HOME}/samples/common/inc

SRCS=$(wildcard *.cxx)
OBJS=$(SRCS:.cxx=)
all: $(OBJS)

$(OBJS): %: %.cxx
	$(CXX) $(INCL) $(CXXFLAGS) $< -o $@

clean:
	rm -f *.o *.lst $(OBJS) *.ptx *.cub
