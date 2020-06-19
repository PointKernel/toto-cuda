module purge
module use /project/projectdirs/m1759/csdaley/Modules/cgpu/modulefiles
module load PrgEnv-llvm/11.0.0-git_20200225
make clean && make
