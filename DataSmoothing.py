import numpy as np
import pylab as plt
import statsmodels.api as sm

# +
count = [0,0,0]

def smoothing(arr_origin,index = 0, weight=0, arr_convert = None):
    if len(arr_origin)>0:
        length =len(arr_origin)
        arr_convert = np.array([[a[0] + (i-length/2)/10000*weight, a[1] + (i-length/2)/10000*weight] for i, a in enumerate(arr_origin)])
#         arr[...,0] += [i/10000*weight for i in range(len(arr))]
#         arr[...,1] += [i/10000*weight for i in range(len(arr))]

        arr_convert = sm.nonparametric.lowess(arr_convert[...,1] , arr_convert[...,0], missing='none')
        nan_check = np.isnan(arr_convert)
        
        if (True in nan_check):
            if weight > 0.6:
                count[index] +=1
                return arr_origin
            
            arr_convert = smoothing(arr_origin,index, weight+0.2, arr_convert)
            
            nan_check = np.isnan(arr_convert)
        if (True in nan_check):
            count[index] +=1
            return arr_origin

        return arr_convert
    return arr_origin


# -

if not [1,2,3]:
    print("AAA")

from tqdm import tqdm
import numpy as np
import pickle
import os

dir_arr = ["val","test","train"]


len(os.listdir("data/argoverse_processed/test"))

for index in range(3):
    pickle_dir = f"data/argoverse_processed/{dir_arr[index]}/"
    save_dir = f"data/argoverse_processed_smoothing/{dir_arr[index]}/"
    file_list = sorted(os.listdir(pickle_dir), key = lambda a: int(a.split(".")[0]))
    print(save_dir)
    for file_index in tqdm(range(len(file_list))):
        with open(pickle_dir + file_list[file_index], 'rb') as f:
            data = pickle.load(f)
        write_pickle = {}
        write_pickle["AGENT"] = {}
        write_pickle["SOCIAL"] = []
        write_pickle['AGENT']["XY_FEATURES"] = smoothing(data['AGENT']["XY_FEATURES"],index)
        if np.any(write_pickle['AGENT']["XY_FEATURES"]) == [False] and len(writeclear_pickle['AGENT']["XY_FEATURES"]) != 0:
            write_pickle['AGENT']["XY_FEATURES"] = data['AGENT']["XY_FEATURES"]
            count[index] +=1
                
        
        
        write_pickle['AGENT']["HEURISTIC_ORACLE_CENTERLINE_NORMALIZED_PARTIAL"] = data['AGENT']["HEURISTIC_ORACLE_CENTERLINE_NORMALIZED_PARTIAL"]
        if "LABELS" in data["AGENT"]:
            write_pickle['AGENT']["LABELS"] = smoothing(data['AGENT']["LABELS"],index)
            if np.any(write_pickle['AGENT']["LABELS"])  == [False] and len(write_pickle['AGENT']["XY_FEATURES"]) != 0:
                write_pickle['AGENT']["LABELS"] = data['AGENT']["LABELS"]
                count[index] +=1
            
            
        write_pickle["PATH"] = data["PATH"]
        write_pickle["SEQ_ID"] = data["SEQ_ID"]
        write_pickle["TRANSLATION"] = data["TRANSLATION"]
        write_pickle["ROTATION"] = data["ROTATION"]
        write_pickle["CITY_NAME"] = data["CITY_NAME"]

        for i in range(len(data['SOCIAL'])):
            T = {}
            T["XY_FEATURES"] = smoothing(data['SOCIAL'][i]["XY_FEATURES"],index)
            if np.any(T["XY_FEATURES"])  == [False] and len(write_pickle['AGENT']["XY_FEATURES"]) != 0:
                T["XY_FEATURES"] = data['SOCIAL'][i]["XY_FEATURES"]
                count[index] +=1

            
            T["HEURISTIC_ORACLE_CENTERLINE_NORMALIZED_PARTIAL"] = data['SOCIAL'][i]["HEURISTIC_ORACLE_CENTERLINE_NORMALIZED_PARTIAL"]
            if "LABELS" in data['SOCIAL'][i]:
                T["LABELS"] = smoothing(data['SOCIAL'][i]["LABELS"],index)
                if np.any(T["LABELS"])  == [False] and len(T["LABELS"]) != 0:
                    T["LABELS"] = data['SOCIAL'][i]["LABELS"]
                    count[index] +=1                
                
            T["TSTAMPS"] = data["SOCIAL"][i]["TSTAMPS"]
            write_pickle["SOCIAL"].append(T)
        with open(save_dir + file_list[file_index], 'wb') as fw:
            pickle.dump(write_pickle, fw)
    print(count[index])
with open("results.txt", "w") as f:
    f.write(str(count))
