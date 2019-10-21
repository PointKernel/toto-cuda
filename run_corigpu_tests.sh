#!/bin/bash

MATIRX=4096
while [  $MATIRX -lt 8192 ]; do
  srun nvprof --metrics flop_count_hp,flop_count_sp	./simpleGEMM $MATIRX
  srun nv-nsight-cu-cli -f --metrics \
  sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hfma_pred_on.sum\
  ./simpleGEMM $MATIRX

  srun nvprof --metrics gld_transactions,\
gst_transactions,\
atomic_transactions,\
local_load_transactions,\
local_store_transactions,\
shared_load_transactions,\
shared_load_transactions,\
l2_read_transactions,\
l2_write_transactions,\
dram_read_transactions,\
dram_write_transactions,\
system_read_transactions,\
system_write_transactions\
  ./simpleGEMM $MATIRX
  srun nv-nsight-cu-cli -f --metrics \
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
dram__sectors_read.sum,\
dram__sectors_write.sum,\
lts__t_sectors_aperture_sysmem_op_read.sum,\
lts__t_sectors_aperture_sysmem_op_write.sum\
  ./simpleGEMM $MATIRX
  let MATIRX=MATIRX*2
done
