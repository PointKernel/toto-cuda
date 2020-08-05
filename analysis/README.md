# Roofline Analysis with Nsight

## Prerequisite
#### `CUDA 10.2 or plus`

## Collect Roofline Data

### Run Nsight profiling on Cori
```bash
srun ./profiling.sh
```

### Convert Nsight output to csv
```bash
nv-nsight-cu-cli --csv -i profiling.nsight-cuprof-report > metrics.csv
```

## Roofline Analysis

1. [ ] `process_nsight.ipynb`: process raw profiling data, compute FLOPS & AI for Roofline plot
2. [ ]  `roofline_plot.ipynb`: plot L1, L2 and HBM Roofline for individual kernels & Hierarchical Roofline for the whole application
