#!/bin/bash
  
srun nv-nsight-cu-cli -f --metrics \
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed,\
smsp__cycles_elapsed.avg.per_second,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__cycles_elapsed.sum,\
smsp__cycles_elapsed.sum.per_second,\
smsp__pipe_tensor_op_hmma_cycles_active.sum,\
smsp__pipe_tensor_op_hmma_cycles_active.sum.per_second\
 ./simpleGEMM
