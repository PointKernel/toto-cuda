import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys, math
import os
import pandas as pd
import matplotlib.patches as mpatches

datadir = os.getcwd()
combineddf = pd.read_csv(os.path.join(datadir,"combined_profile.csv"))

#parameters
plotdir = "./plots"
font = { 'size'   : 15}
plt.rc('font', **font)
markersize = 10 #12

#markers
colors = ['b','r','g','m','y','c']
styles = ['o','s','v','^','D',">","<","*","h","H","+","1","2","3","4","8","p","d","|","_",".",","]

#styledict
styledict = {"thorsten": {"fontsize_annotation": 10, "roof_color": 'gray', "legend_points_ncol": 2, "frameon": False},
             "charlene": {"fontsize_annotation": 15, "roof_color": 'k', "legend_points_ncol": 1, "frameon": True}}

#data dependent stuff
def plot_data(file_prefix, plot_label, marker_tag, marker_label, df):
    
    #pick style:
    style = styledict["thorsten"]
    fontsize_annotation = style["fontsize_annotation"]

    print(df.columns)

    df[['FLOPs Avg', 'L1 Bytes Avg', 'L2 Bytes Avg', 'DRAM Bytes Avg']] /= 1e9
    
    #fp32 data
    df_fp32         = df[ df["Precision"]=="FP32" ]
    print(df_fp32[['L1 Bytes Avg', 'L2 Bytes Avg', 'DRAM Bytes Avg', 'FLOPs Avg', 'FP32 FLOPs Avg',
        'FP16 FLOPs Avg', 'CUDA Time Avg', 'TC Time Avg']])
    BYTES_L1_fp32   = list(df_fp32["L1 Bytes Avg"])
    BYTES_L2_fp32   = list(df_fp32["L2 Bytes Avg"])
    BYTES_DRAM_fp32 = list(df_fp32["DRAM Bytes Avg"])
    FLOPS_fp32      = list(df_fp32["FLOPs Avg"])
    print(FLOPS_fp32)
    TIME_fp32       = list(df_fp32["CUDA Time Avg"])
    labels_fp32 = ["FP32 "+marker_label+" "+str(x) for x in list(df_fp32[marker_tag])]

    #fp16 data
    df_fp16         = df[ df["Precision"]=="FP16" ]
    BYTES_L1_fp16   = list(df_fp16["L1 Bytes Avg"])
    BYTES_L2_fp16   = list(df_fp16["L2 Bytes Avg"])
    BYTES_DRAM_fp16 = list(df_fp16["DRAM Bytes Avg"])
    FLOPS_fp16      = list(df_fp16["FLOPs Avg"])
    TIME_fp16       = list(df_fp16["CUDA Time Avg"])
    labels_fp16 = ["FP16 "+marker_label+" "+str(x) for x in list(df_fp16[marker_tag])]
    
    #figure stuff
    fig = plt.figure(1,figsize=(10.67,6.6))
    plt.clf()
    ax = fig.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('GBytes')
    ax.set_ylabel('GFLOPs')

    plt.grid(True, which="major", ls="--", lw=1)
    plt.grid(True, which="minor", ls="--", lw=0.5)

    xmin = min(min(BYTES_L1_fp16), min(BYTES_L1_fp32), min(BYTES_L2_fp16), min(BYTES_L2_fp32), min(BYTES_DRAM_fp16), min(BYTES_DRAM_fp32))
    xmax = max(max(BYTES_L1_fp16), max(BYTES_L1_fp32), max(BYTES_L2_fp16), max(BYTES_L2_fp32), max(BYTES_DRAM_fp16), max(BYTES_DRAM_fp32))
    ymin = min(min(FLOPS_fp16), min(FLOPS_fp32))
    ymax = max(max(FLOPS_fp16), max(FLOPS_fp32))
    zmax = max(max(TIME_fp16), max(TIME_fp32))
    xmax = pow(10, math.log(xmax)//math.log(10) + 2)
    ymax = pow(10, math.log(ymax)//math.log(10) + 3)
    zmax = round(zmax, 1)

    overhead = 4.2e-6

    #mem
    smemroofs = [828.758, 2996.77, 14*1e3]
    smem_roof_name = ['HBM', 'L2', 'L1']

    #flops
    scomproofs = [15158.23, 29181.64, 125000.0]
    scomp_roof_name = ['FP32 (FMA)', 'FP16 (FMA)', 'Tensor Core']
    
    oh_memroofs = []
    for i in range(len(smemroofs)):
        oh_memroofs.append(smemroofs[i] * overhead)

    oh_comproofs = []
    for i in range(len(scomproofs)):
        oh_comproofs.append(scomproofs[i] * overhead)

    # plot memory ceilings
    for i, element in enumerate(oh_memroofs):
        ax.plot([element, element], [0, oh_comproofs[-1]], c=colors[i], linestyle='dashed')
    # plot flop ceilings
    for element in oh_comproofs:
        ax.plot([0, oh_memroofs[-1]], [element, element], c='k', linestyle='dashed')

    xmin = pow(10, math.log(oh_memroofs[0])//math.log(10))
    ymin = pow(10, math.log(oh_comproofs[0])//math.log(10))

    #some handles
    marker_handles = []
    
    #plot roofs
    #plot_roofs(fig, zmax, ax, style)

    #FP32
    #for i in range(len(BYTES_L1_fp32)):
    #    ax.plot(float(BYTES_L1_fp32[i]),float(FLOPS_fp32[i]), c=colors[0],marker=styles[i], mfc='none',linestyle='None',ms=markersize,label=labels_fp32[i])
    #ax.plot((BYTES_L1_fp32),(FLOPS_fp32),c=colors[0],linestyle='-')
    #for i in range(len(BYTES_L2_fp32)):
    #    ax.plot(float(BYTES_L2_fp32[i]),float(FLOPS_fp32[i]),c=colors[1],marker=styles[i], mfc='none',linestyle='None',ms=markersize,label=labels_fp32[i])
    #ax.plot((BYTES_L2_fp32),(FLOPS_fp32),c=colors[1],linestyle='-')
    #for i in range(len(BYTES_DRAM_fp32)):
    #    ax.plot(float(BYTES_DRAM_fp32[i]),float(FLOPS_fp32[i]),c=colors[2],marker=styles[i], mfc='none', linestyle='dashed',ms=markersize,label=labels_fp32[i])
    #    ax.annotate(str('%.2f' % TIME_fp32[i]), (BYTES_DRAM_fp32[i], FLOPS_fp32[i]), color='k')
    #    marker_handles.append(ax.plot([],[],c='gray',marker=styles[i], mfc='none', linestyle='None',ms=markersize,label=labels_fp32[i])[0])
    #ax.plot((BYTES_DRAM_fp32),(FLOPS_fp32),c=colors[2],linestyle='-')
    i = 0 
    ax.plot(float(BYTES_L1_fp32[i]),float(FLOPS_fp32[i]), c=colors[0],marker=styles[i], mfc='none',linestyle='None',ms=markersize,label=labels_fp32[i])
    ax.plot(float(BYTES_L2_fp32[i]),float(FLOPS_fp32[i]),c=colors[1],marker=styles[i], mfc='none',linestyle='None',ms=markersize,label=labels_fp32[i])
    ax.plot(float(BYTES_DRAM_fp32[i]),float(FLOPS_fp32[i]),c=colors[2],marker=styles[i], mfc='none', linestyle='dashed',ms=markersize,label=labels_fp32[i])
    ax.annotate(str('%.2f' % TIME_fp32[i]), (BYTES_DRAM_fp32[i], FLOPS_fp32[i]), color='k', horizontalalignment='right')

    flop_ceilings = []
    for element in scomproofs:
        flop_ceilings.append(TIME_fp32[i] * element)

    byte_ceilings = []
    for element in smemroofs:
        byte_ceilings.append(TIME_fp32[i] * element)

    print(flop_ceilings)
    print(byte_ceilings)

    # plot memory ceilings
    for i, e in enumerate(byte_ceilings):
        ax.plot([e, e], [0, flop_ceilings[-1]], c=colors[i])

    # plot flop ceilings
    for i, e in enumerate(flop_ceilings):
        ax.plot([0, byte_ceilings[-1]], [e, e], c='k')
        label = scomp_roof_name[i] + ': ' + '{0:.1f}'.format(float(e))
        ax.annotate(label, xy=(xmin,e), xytext=(17,5), textcoords="offset points", color='k', horizontalalignment='left', fontsize=fontsize_annotation)

    xmax = pow(10, math.log(byte_ceilings[-1])//math.log(10) + 1)
    ymax = pow(10, math.log(flop_ceilings[-1])//math.log(10) + 2)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # plot 
    for k in range(len(oh_memroofs)):
        ax.plot([oh_memroofs[k], byte_ceilings[k]], [oh_comproofs[-1], flop_ceilings[-1]], c=colors[k])

    #FP16
    #for i in range(0,len(AI_L1_fp16)):
    #    ax.plot(float(AI_L1_fp16[i]),float(FLOPs_fp16[i]),c=colors[0],marker=styles[i], linestyle='None',ms=markersize,label=labels_fp16[i])
    #ax.plot((AI_L1_fp16),(FLOPs_fp16),c=colors[0],linestyle='-')
    #for i in range(0,len(AI_L2_fp16)):
    #    ax.plot(float(AI_L2_fp16[i]),float(FLOPs_fp16[i]),c=colors[1],marker=styles[i], linestyle='None',ms=markersize,label=labels_fp16[i])
    #ax.plot((AI_L2_fp16),(FLOPs_fp16),c=colors[1],linestyle='-')
    #for i in range(0,len(AI_DRAM_fp16)):
    #    ax.plot(float(AI_DRAM_fp16[i]),float(FLOPs_fp16[i]),c=colors[2],marker=styles[i], linestyle='dashed',ms=markersize,label=labels_fp16[i])
    #    marker_handles.append(ax.plot([],[],c='gray',marker=styles[i], linestyle='None',ms=markersize,label=labels_fp16[i])[0])
    #ax.plot((AI_DRAM_fp16),(FLOPs_fp16),c=colors[2],linestyle='-')
    
    #annotations
    #legend 1
    leg1 = plt.legend(handles = marker_handles, loc="upper right", ncol=style["legend_points_ncol"], frameon=style["frameon"])
    ax.add_artist(leg1)
    
    #legend 2:
    patch_handles = []
    for i in range(len(smem_roof_name)):
        patch_handles.append(mpatches.Patch(color=colors[i],label = smem_roof_name[i]))
    leg2 = plt.legend(handles = patch_handles,loc='upper left', scatterpoints = 1, frameon=style["frameon"])
    
    #title
    #if plot_label:
    #    ax.text(ax.get_xlim()[0]*1.1, ax.get_ylim()[1]*1.5, plot_label, horizontalalignment='left', verticalalignment='top')
   
    #plt.show()

    #save figure
    plt.tight_layout()
    plt.savefig(file_prefix+'.png')
    plt.savefig(file_prefix+'.eps')


#plot vs batchsize
directory=os.getcwd()
try:
    os.stat(directory)
except:
    os.mkdir(directory)

features = list(combineddf[["Network Name", "Input Shape", "Kernel Shape", "Stride Size", "Pass"]].apply(lambda x: (x["Network Name"], \
                                                                                          x["Input Shape"], \
                                                                                          x["Kernel Shape"], \
                                                                                          x["Stride Size"], \
                                                                                          x["Pass"]), axis=1).unique())

for idx,feature in enumerate(features):
    
    #project the data
    selectdf = combineddf[ (combineddf[ "Network Name" ] == feature[0]) & \
                           (combineddf[ "Input Shape" ] == feature[1]) & \
                           (combineddf[ "Kernel Shape" ] == feature[2]) & \
                           (combineddf[ "Stride Size" ] == feature[3]) & \
                           (combineddf[ "Pass" ] == feature[4]) ]
    
    if len(selectdf["Batch Size"].unique()) == 1:
        continue
    
    #label
    plot_label = 'hierarchical_'+feature[4]
    plot_file = os.path.join(directory, plot_label.replace(", ","_"))
    plot_label = None
    plot_data(plot_file, plot_label, "Batch Size", "batch size", selectdf)
