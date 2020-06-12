#!/bin/bash

### Tensor Core utilization
metrics="\
sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active,"

### FLOP
# DP
metrics+="\
smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,"
# SP
metrics+="\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,"
# HP
metrics+="\
smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,"

### Time
# CUDA Core time
metrics+="\
smsp__cycles_elapsed.sum,\
smsp__cycles_elapsed.sum.per_second,"
# Tensor Core time
metrics+="\
smsp__pipe_tensor_op_hmma_cycles_active.sum,\
smsp__pipe_tensor_op_hmma_cycles_active.sum.per_second,"

### L1 transactions
# local
metrics+="\
l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,"
# shared
metrics+="\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,"
# global
metrics+="\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,"
# atomic
metrics+="\
l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum,\
l1tex__t_set_accesses_pipe_lsu_mem_global_op_red.sum,\
l1tex__t_set_accesses_pipe_tex_mem_surface_op_atom.sum,\
l1tex__t_set_accesses_pipe_tex_mem_surface_op_red.sum,"

### L2 transactions
# read + write
metrics+="\
lts__t_sectors_op_read.sum,\
lts__t_sectors_op_write.sum,"
#atomic
metrics+="\
lts__t_sectors_op_atom.sum,\
lts__t_sectors_op_red.sum,"

### HBM transactions
metrics+="\
dram__sectors_read.sum,\
dram__sectors_write.sum,"

### PCI/NVLINK transactions
metrics+="\
lts__t_sectors_aperture_sysmem_op_read.sum,\
lts__t_sectors_aperture_sysmem_op_write.sum"


profilestring="nv-nsight-cu-cli --target-processes all --metrics ${metrics} -f -o profiling"

${profilestring} ./app
