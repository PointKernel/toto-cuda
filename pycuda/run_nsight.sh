#!/bin/bash
  
srun nv-nsight-cu-cli --profile-from-start off -f --csv --metrics \
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
dram__sectors_read.sum,\
dram__sectors_write.sum\
 python test.py
