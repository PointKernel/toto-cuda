#!/bin/bash

srun nv-nsight-cu-cli -f --metrics \
sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,\
smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,\
smsp__cycles_elapsed.sum,\
smsp__cycles_elapsed.sum.per_second,\
smsp__pipe_tensor_op_hmma_cycles_active.sum,\
smsp__pipe_tensor_op_hmma_cycles_active.sum.per_second,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum,\
l1tex__t_set_accesses_pipe_lsu_mem_global_op_red.sum,\
l1tex__t_set_accesses_pipe_tex_mem_surface_op_atom.sum,\
l1tex__t_set_accesses_pipe_tex_mem_surface_op_red.sum,\
l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,\
lts__t_sectors_op_read.sum,\
lts__t_sectors_op_write.sum,\
lts__t_sectors_op_atom.sum,\
lts__t_sectors_op_red.sum,\
dram__sectors_read.sum,\
dram__sectors_write.sum\
  ./simpleGEMM


srun nvprof -f --metrics \
l2_read_transactions,\
l2_write_transactions,\
dram_write_transactions,\
dram_read_transactions,\
gld_transactions,\
gst_transactions,\
atomic_transactions,\
local_load_transactions,\
local_store_transactions,\
shared_load_transactions,\
shared_store_transactions,\
flop_count_hp,\
flop_count_sp,\
flop_count_dp\
  ./simpleGEMM
