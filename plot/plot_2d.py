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

    #print(df.columns)

    df[['FP16 FLOPs Avg']] /= 2
    df['FLOPs Avg'] = df['FP16 FLOPs Avg'] + df['FP32 FLOPs Avg'] + df['TC FLOPs Avg']
    df[['FLOPs Avg', 'L1 Bytes Avg', 'L2 Bytes Avg', 'DRAM Bytes Avg']] /= 1e9
    df[['CUDA Time Avg', 'TC Time Avg']] *= 1e3
    #print(df[['L1 Bytes Avg', 'L2 Bytes Avg', 'DRAM Bytes Avg', 'FLOPs Avg', 'FP32 FLOPs Avg',
    #    'FP16 FLOPs Avg', 'TC FLOPs Avg', 'CUDA Time Avg', 'TC Time Avg']])
    #print(df)

    #fp16 data
    BYTES_L1   = list(df["L1 Bytes Avg"])
    BYTES_L2   = list(df["L2 Bytes Avg"])
    BYTES_DRAM = list(df["DRAM Bytes Avg"])
    FLOPS      = list(df["FLOPs Avg"])
    TIME_CUDA  = list(df["CUDA Time Avg"])
    TIME_TC    = list(df["TC Time Avg"])
    labels     = ["FP16 "+marker_label+" "+str(x) for x in list(df[marker_tag])]
    datasize   = len(FLOPS)
    
    #figure stuff
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7.5))
    #fig.suptitle('Performance analysis')
    #ax1.plot(x, y)
    #ax2.plot(x, -y)
    #fig = plt.figure(1,figsize=(10.67,6.6))
    #plt.clf()
    #ax1 = fig.gca()
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('GBytes')
    ax1.set_ylabel('GFLOPs')

    ax1.grid(True, which="major", ls="--", lw=1)
    ax1.grid(True, which="minor", ls="--", lw=0.5)

    bmin = min(min(BYTES_L1), min(BYTES_L2), min(BYTES_DRAM))
    bmax = max(max(BYTES_L1), max(BYTES_L2), max(BYTES_DRAM))
    fmin = min(FLOPS)
    fmax = max(FLOPS)
    tmin = min(TIME_CUDA)
    tmax = max(TIME_CUDA)

    bmax = pow(10, math.log(bmax)//math.log(10) + 2)
    fmax = pow(10, math.log(fmax)//math.log(10) + 3)
    tmax = round(tmax, 1)

    # kernel launch overhead is ~ 4.2 us
    overhead = 4.2e-6

    #mem
    memroof      = 828.758
    memroof_name = 'HBM'

    #flops
    fp32roof = 15158.23
    tcroof   = 125000.0
    
    # overhead ceilings
    oh_byteroof = memroof * overhead
    oh_floproof = 0
    oh_floproof += fp32roof * overhead
    oh_floproof += tcroof * overhead

    # plot memory ceilings
    ax1.plot([oh_byteroof, oh_byteroof], [0, oh_floproof], c=colors[-1])
    # plot flop ceilings
    ax1.plot([0, oh_byteroof], [oh_floproof, oh_floproof], c=colors[-1])

    bmin = pow(10, math.log(oh_byteroof)//math.log(10))
    fmin = pow(10, math.log(oh_floproof)//math.log(10))
    ax1.annotate("Overhead-bound", xy=(bmin, fmin), xytext=(17,5), textcoords="offset points",
            color=colors[-1], horizontalalignment='left', fontsize=fontsize_annotation,
            verticalalignment='bottom', rotation=270)

    #some handles
    marker_handles = []
    
    old_byte_ceiling, old_flop_ceiling = oh_byteroof, oh_floproof
    for i in range(datasize):
        flop_ceiling = TIME_CUDA[i] * fp32roof / 1e3
        flop_ceiling += TIME_TC[i] * tcroof / 1e3
        byte_ceiling = TIME_CUDA[i] * memroof / 1e3
        #print("bytes: {} flops: {}".format(BYTES_DRAM[i], FLOPS[i]))
        ax1.plot(float(BYTES_DRAM[i]),float(FLOPS[i]), c=colors[i], marker=styles[0], mfc='none',linestyle='None', ms=markersize)
        # plot memory ceilings
        ax1.plot([byte_ceiling, byte_ceiling], [0, flop_ceiling], c=colors[i])
        label = 'Bytes ceiling: ' + '{0:.1f}'.format(float(byte_ceiling))
        ax1.annotate(label, xy=(byte_ceiling, fmin), xytext=(17,5), textcoords="offset points",
                color=colors[i], horizontalalignment='right', fontsize=fontsize_annotation,
                verticalalignment='bottom', rotation=270)
        # plot flop ceilings
        ax1.plot([0, byte_ceiling], [flop_ceiling, flop_ceiling], c=colors[i])
        label = 'Mixed-precision GFLOPs ceiling: ' + '{0:.1f}'.format(float(flop_ceiling))
        ax1.annotate(label, xy=(bmin,flop_ceiling), xytext=(17,5), textcoords="offset points",
                color=colors[i], horizontalalignment='left', fontsize=fontsize_annotation)
        # plot 
        ax1.plot([old_byte_ceiling, byte_ceiling], [old_flop_ceiling, flop_ceiling], c='k')

        old_flop_ceiling, old_byte_ceiling = flop_ceiling, byte_ceiling
        #marker_handles.append(ax1.plot([],[],c='gray',marker=styles[i], linestyle='None',ms=markersize,label=labels_fp16[i])[0])
    #ax1.plot((BYTES_DRAM),(FLOPS),c=colors[-1],linestyle='-', linewidth=2)

    bmax = pow(10, math.log(old_byte_ceiling)//math.log(10) + 1)
    fmax = pow(10, math.log(old_flop_ceiling)//math.log(10) + 1)

    ax1.set_xlim(bmin, bmax)
    ax1.set_ylim(fmin, fmax)

    #legend 1
    #leg1 = plt.legend(handles = marker_handles, loc="upper right", ncol=style["legend_points_ncol"], frameon=style["frameon"])
    #ax.add_artist(leg1)
    
    #legend 2:
    patch_handles = []
    for i in range(len(labels)):
        patch_handles.append(mpatches.Patch(color=colors[i],label = labels[i]))
    leg1 = ax1.legend(handles = patch_handles,loc='upper left', scatterpoints = 1,
            frameon=style["frameon"], fontsize = 12)
    leg1 = ax2.legend(handles = patch_handles,loc='upper left', scatterpoints = 1,
            frameon=style["frameon"], fontsize = 12)


    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('GFLOPs')
    ax2.set_ylabel('Runtime [ms]')

    ax2.grid(True, which="major", ls="--", lw=1)
    ax2.grid(True, which="minor", ls="--", lw=0.5)

    # plot kernel runtime
    ax2.plot([0, oh_floproof], [overhead * 1e3 , overhead * 1e3], c=colors[-1])
    # plot flop ceilings
    #ax2.plot([oh_floproof, oh_floproof], [0, overhead * 1e3], c='k', linestyle='dashed')

    fmin = pow(10, math.log(oh_floproof)//math.log(10))
    tmin = pow(10, math.log(overhead * 1e3)//math.log(10))
    ax2.annotate("Overhead limit", xy=(fmin, overhead * 1e3/2), xytext=(17,5), textcoords="offset points",
            color=colors[-1], horizontalalignment='left', fontsize=fontsize_annotation,
            verticalalignment='baseline')

    old_flop_ceiling, old_t_ceiling = oh_floproof, overhead * 1e3
    for i in range(datasize):
        flop_ceiling = TIME_CUDA[i] * fp32roof / 1e3
        flop_ceiling += TIME_TC[i] * tcroof / 1e3
        t_ceiling = TIME_CUDA[i]
        #print("bytes: {} flops: {}".format(BYTES_DRAM[i], FLOPS[i]))
        ax2.plot(float(FLOPS[i]),float(TIME_CUDA[i]), c=colors[i], marker=styles[0], mfc='none',linestyle='None', ms=markersize)
        ax2.annotate(str('%.2f' % TIME_CUDA[i]), (FLOPS[i], TIME_CUDA[i]), color='k',
                horizontalalignment='right', fontsize=fontsize_annotation)
        #ax2.axvline(x=flop_ceiling, c=colors[i], linestyle='dashed')
        #ax2.plot([float(FLOPS[i]), flop_ceiling], [TIME_CUDA[i], TIME_CUDA[i]], c=colors[i],
        #        linestyle='dashed')
        # plot minimum runtime
        ax2.plot([old_flop_ceiling, flop_ceiling], [old_t_ceiling, t_ceiling], c='k')

        slope = (old_t_ceiling - t_ceiling) / (old_flop_ceiling - flop_ceiling)
        min_runtime = slope * (FLOPS[i] - old_flop_ceiling) + old_t_ceiling
        ax2.plot([float(FLOPS[i]), float(FLOPS[i])], [min_runtime, TIME_CUDA[i]], c='k',
                linestyle='dashed')
        ax2.annotate("Min. runtime: " + str('%.2f' % min_runtime), (FLOPS[i], min_runtime), color='k',
                horizontalalignment='left', verticalalignment='top', fontsize=fontsize_annotation)

        old_flop_ceiling, old_t_ceiling = flop_ceiling, t_ceiling
    
    tmax = pow(10, math.log(old_t_ceiling)//math.log(10) + 1)
    fmax = pow(10, math.log(old_flop_ceiling)//math.log(10) + 1)
    ax2.plot([old_flop_ceiling, (tmax - old_t_ceiling)/slope + old_flop_ceiling], [old_t_ceiling, tmax], c='k')

    ax2.set_xlim(fmin, fmax)
    ax2.set_ylim(tmin, tmax)

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
    plot_label = '2d_'+feature[4]
    plot_file = os.path.join(directory, plot_label.replace(", ","_"))
    plot_label = None
    selectdf = selectdf[selectdf['Precision'] == 'FP16']
    plot_data(plot_file, plot_label, "Batch Size", "batch size", selectdf)
