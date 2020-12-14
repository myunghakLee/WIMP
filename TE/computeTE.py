# -*- coding: utf-8 -*-
from sys import platform
import glob
import numpy as np
from src.utils import *
import pickle
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import pandas as pd

import h5py
import os

from tqdm import tqdm

from jpype import *
import random
import math

if __name__ == '__main__':
    dataset = './../data/argoverse_processed/val/'
    config = dict2(**{
        'datapath': f'./{dataset}/',
        'index': ['id', 'x', 'y'],
        'savepath': f'./TEdata/val/'
    })

    if not os.path.isdir(config.datapath):
        exit(f"Data path not found ({config.datapath})")

    if not os.path.isdir(config.savepath):
        os.mkdir(config.savepath)

    #
    # Setting Java for TE compute
    #
    jarLocation = "./infodynamics.jar"
    if not os.path.isfile(jarLocation):
        exit("infodynamics.jar not found (expected at " + os.path.abspath(
            jarLocation) + ") - are you running from demos/python?")
    # Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

    # Create a Kraskov TE calculator:
    teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorMultiVariateKraskov

    teCalc = teCalcClass()

    # Set properties for auto-embedding of both source and destination
    #  using the Ragwitz criteria:
    #  a. Auto-embedding method
    teCalc.setProperty(teCalcClass.PROP_AUTO_EMBED_METHOD, teCalcClass.AUTO_EMBED_METHOD_RAGWITZ)
    #  b. Search range for embedding dimension (k) and delay (tau)
    teCalc.setProperty(teCalcClass.PROP_K_SEARCH_MAX, "6")
    teCalc.setProperty(teCalcClass.PROP_TAU_SEARCH_MAX, "6")

    # Since we're auto-embedding, no need to supply k, l, k_tau, l_tau here:
    teCalc.initialise(2, 2)

    scenarios = sorted(glob.glob(config.datapath + '*'))

    for scenario in tqdm(scenarios):
        label_name = scenario.split('/')[-1].split('.')[0]

        try:
            label = pd.read_pickle(scenario)
        except:
            continue
        agent = label['AGENT']
        others = label['SOCIAL']
        num_others = len(others)

        te = np.zeros((num_others, 2))
        data_source = np.concatenate((agent['XY_FEATURES'],agent['LABELS']), axis=0)
        for idx_desc in range(num_others):
            if len(others[idx_desc]['LABELS'] > 0):
                data_desc = np.concatenate((others[idx_desc]['XY_FEATURES'],others[idx_desc]['LABELS']), axis=0)
            else:
                data_desc = others[idx_desc]['XY_FEATURES']
            max_time = min(len(data_desc), len(data_source))

            ind_overlap = np.logical_and(np.prod(data_source[:max_time], axis=1), np.prod(data_desc[:max_time], axis=1))
            nOverlap = np.sum(ind_overlap)
            if nOverlap >= 8:
                ind2interest = np.where(ind_overlap == True)

                data_source = data_source[ind2interest]
                data_desc = data_desc[ind2interest]

                # Compute TE
                if len(data_source.tolist()) > 40 and len(data_desc.tolist()) > 40:
                    teCalc.setObservations(data_source.tolist(), data_desc.tolist())
                    teSourceToDesc = teCalc.computeAverageLocalOfObservations()

                    teCalc.setObservations(data_desc.tolist(), data_source.tolist())
                    teDescToSource = teCalc.computeAverageLocalOfObservations()
                else:
                    teSourceToDesc = None
                    teDescToSource = None
                te[idx_desc][0] = teSourceToDesc
                te[idx_desc][1] = teDescToSource

        with open(f'{config.savepath}/{label_name}.pkl', 'wb') as f:
            pickle.dump(te, f, pickle.HIGHEST_PROTOCOL)

# +
# import random as r
# [[r.random()*10,r.random()*10] for i in range(20)]

# +
# 5초간의 데이터

# agent['XY_FEATURES'] --> 처음 2초
# agent['LABELS']      --> 그 다음 3초

