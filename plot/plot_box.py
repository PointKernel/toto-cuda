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

    df[['FLOPs Avg', 'L1 Bytes Avg', 'L2 Bytes Avg', 'DRAM Bytes Avg']] /= 1e9
    df[['CUDA Time Avg', 'TC Time Avg']] *= 1e3

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
    fig = plt.figure(1, figsize=(10, 10))
    plt.clf()
    ax = fig.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('GBytes')
    ax.set_ylabel('GFLOPs')

    ax.grid(True, which="major", ls="--", lw=1)
    ax.grid(True, which="minor", ls="--", lw=0.5)

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
    byteroofs = [828.8, 2996.7, 14*1024]
    byteroofnames = ['HBM', 'L2', 'L1']

    #flops
    floproofs = [7669.1, 15158.2, 29181.6, 125000.0]
    floproofnames = ['FP64', 'FP32', 'FP16', 'TC']
    
    # overhead ceilings
    oh_byteroof = byteroofs[0] * overhead
    oh_floproof = floproofs[0] * overhead

    # plot memory ceilings
    ax.plot([oh_byteroof, oh_byteroof], [0, oh_floproof], c=colors[-1])
    # plot flop ceilings
    ax.plot([0, oh_byteroof], [oh_floproof, oh_floproof], c=colors[-1])

    bmin = pow(10, math.log(oh_byteroof)//math.log(10))
    fmin = pow(10, math.log(oh_floproof)//math.log(10))
    ax.annotate("Overhead-bound", xy=(bmin, fmin), xytext=(5,10), textcoords="offset points",
            color=colors[-1], horizontalalignment='left', fontsize=fontsize_annotation,
            verticalalignment='bottom')

    #some handles
    marker_handles = []
    
    old_byte_ceiling, old_flop_ceiling = oh_byteroof, oh_floproof
    for i in range(datasize):

        ax.plot(float(BYTES_DRAM[i]),float(FLOPS[i]), c=colors[i], marker=styles[0], mfc='none',linestyle='None', ms=markersize)

        for j, flop in enumerate(floproofs):
            flop_ceiling = TIME_CUDA[i] * flop / 1e3
            byte_ceiling = TIME_CUDA[i] * byteroofs[0] / 1e3

            if flop_ceiling > FLOPS[i]:
                # plot memory ceilings
                ax.plot([byte_ceiling, byte_ceiling], [FLOPS[i], flop_ceiling], c=colors[i])
                #label = 'Bytes ceiling: ' + '{0:.1f}'.format(float(byte_ceiling))
                #ax.annotate(label, xy=(byte_ceiling, fmin), xytext=(17,5), textcoords="offset points",
                #        color=colors[i], horizontalalignment='right', fontsize=fontsize_annotation,
                #        verticalalignment='bottom', rotation=270)
                # plot flop ceilings
                ax.plot([BYTES_DRAM[i], byte_ceiling], [flop_ceiling, flop_ceiling], c=colors[i])
                label = floproofnames[j] #+ ' {0:.1f}'.format(float(flop_ceiling))
                ax.annotate(label, xy=(BYTES_DRAM[i], flop_ceiling), xytext=(3,3), textcoords="offset points",
                        color=colors[i], horizontalalignment='left', fontsize=fontsize_annotation)
                ax.plot([BYTES_DRAM[i], byte_ceiling], [FLOPS[i], FLOPS[i]], c=colors[i], linestyle='--')
                ax.plot([BYTES_DRAM[i], BYTES_DRAM[i]], [FLOPS[i], flop_ceiling], c=colors[i], linestyle='--')
                # plot 

                old_flop_ceiling, old_byte_ceiling = flop_ceiling, byte_ceiling
                #marker_handles.append(ax.plot([],[],c='gray',marker=styles[i], linestyle='None',ms=markersize,label=labels_fp16[i])[0])
                #ax.plot((BYTES_DRAM),(FLOPS),c=colors[-1],linestyle='-', linewidth=2)

    ax.plot([oh_byteroof, old_byte_ceiling], [oh_floproof, old_flop_ceiling], c='k')

    bmax = pow(10, math.log(old_byte_ceiling)//math.log(10) + 1)
    fmax = pow(10, math.log(old_flop_ceiling)//math.log(10) + 1)

    ax.set_xlim(bmin, bmax)
    ax.set_ylim(fmin, fmax)

    #legend 1
    #leg1 = plt.legend(handles = marker_handles, loc="upper right", ncol=style["legend_points_ncol"], frameon=style["frameon"])
    #ax.add_artist(leg1)
    
    #legend 2:
    patch_handles = []
    for i in range(len(labels)):
        patch_handles.append(mpatches.Patch(color=colors[i],label = labels[i]))
    leg1 = ax.legend(handles = patch_handles,loc='upper left', scatterpoints = 1,
            frameon=style["frameon"], fontsize = 12)

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
    plot_label = 'box_'+feature[4]
    plot_file = os.path.join(directory, plot_label.replace(", ","_"))
    plot_label = None
    selectdf = selectdf[selectdf['Precision'] == 'FP16']
    plot_data(plot_file, plot_label, "Batch Size", "batch size", selectdf)
