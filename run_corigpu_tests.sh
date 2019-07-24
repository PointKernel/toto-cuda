#!/bin/bash

MATIRX=128
while [  $MATIRX -lt 65536 ]; do
  srun nvprof --print-gpu-summary ./simpleGEMM $MATIRX
  srun nv-nsight-cu-cli -f --metrics sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active ./simpleGEMM $MATIRX
  let MATIRX=MATIRX*2
done
