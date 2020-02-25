#!/bin/bash

MATRIX=128
while [  $MATRIX -lt 65536 ]; do
  srun nvprof --print-gpu-summary ./simpleGEMM $MATRIX
  srun nv-nsight-cu-cli -f --metrics sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active ./simpleGEMM $MATRIX
  let MATRIX=MATRIX*2
done
