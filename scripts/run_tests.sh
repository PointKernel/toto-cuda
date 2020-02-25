#!/bin/bash

srun nv-nsight-cu-cli -o res_nsight -f --metrics \
sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__cycles_elapsed.sum,\
smsp__cycles_elapsed.sum.per_second,\
lts__t_sectors_op_read.sum,\
lts__t_sectors_op_write.sum,\
lts__t_sectors_op_atom.sum,\
lts__t_sectors_op_red.sum,\
dram__sectors_read.sum,\
dram__sectors_write.sum\
  ./simpleGEMM


srun nvprof -o res_nvprof -f --metrics \
tensor_precision_fu_utilization,\
l2_read_transactions,\
l2_write_transactions,\
dram_write_transactions,\
dram_read_transactions,\
flop_count_sp\
  ./simpleGEMM
