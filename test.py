import subprocess as sp
import numpy as np
import pandas as pd
from io import StringIO
import os
import re
import shutil
import sqlite3

def import_nsight_metric(filename, cuda_dir='/usr/common/software/cuda/10.2.89/'):
    #execute nvprof and parse file
    args = [os.path.join(cuda_dir, "bin/nv-nsight-cu-cli"),"--csv","-i",filename]
    #skiprows = 2
        
    #open subprocess and communicate
    p = sp.Popen(args, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = p.communicate()
    
    #get timeline from csv
    profiledf = pd.read_csv(StringIO(stdout.decode("utf-8")),skiprows=0) #.dropna(how="all").rename(columns={"Kernel": "Name"})
    
    #clean up
    del profiledf["Process ID"]
    del profiledf["Process Name"]
    del profiledf["Host Name"]
    del profiledf["Kernel Time"]
    del profiledf["Context"]
    del profiledf["Stream"]
    del profiledf["Section Name"]
    
    profiledf = profiledf.groupby(["Kernel Name", "Metric Name"]).apply(lambda x: pd.Series([x["Metric Value"].count(),x["Metric Value"].sum()])).reset_index()
    profiledf.rename(columns={0: "Invocations", 1: "Metric Value", "Kernel Name": "Name"}, inplace=True)
    
    #return result
    return profiledf


filename = "res_test.nsight-cuprof-report"

print(import_nsight_metric(filename))
