# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 11:34:08 2019

@author: Ahmert
"""

import os
import numpy
import matplotlib.pyplot as plt
from collections import Counter

def show_frequencies(arry):
    arry.sort()
    freqs = Counter(arry)
    
    labels, values = zip(*freqs.items())
    
    indexes = numpy.arange(len(labels))
    width = 1
    
    print(labels)
    print(values)
    
    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels)
    plt.show()

def get_str(line):
    oline = ''
    for wor in line:
        oline += wor
        oline += ' '
    return oline

# find folders
NODE_COUNTS = [500]
#P_INS = [0.04, 0.05, 0.06, 0.068, 0.075, 0.09]
#P_OUTS = [0.008, 0.011, 0.0125, 0.015, 0.016, 0.018]
P_INS = [0.05, 0.075, 0.09]
P_OUTS = [0.011, 0.016, 0.018]
#P_INS = [0.075]
#P_OUTS = [0.016]
GAUS_UPS = [30]
GAUS_DOWNS = [18]
ALPHAS = [0.33,0.5,0.75]
#ALPHAS = [0.5]
sub_names = ['EFF_GRPATT']

step_lens = []
step_step_lens = []
all_step_lens = []

savedirectory = './GR_TEST'
file_indiv = open(savedirectory + '/indiv_summ.stxt', "w")

for NODE_COUNT in NODE_COUNTS:
    for GAUS_UP in GAUS_UPS:
        for GAUS_DOWN in GAUS_DOWNS:
            for P_IN in P_INS:
                for P_OUT in P_OUTS:
                    for ALPHA in ALPHAS:
                        for sub_name in sub_names:

                            directory = './GR_TEST/NODE_' + str(NODE_COUNT) + '/DENS_' + str(GAUS_UP) + '_' + str(GAUS_DOWN) + '_' + \
                                str(P_IN) + '_' + str(P_OUT) + '/ALPHA_' + str(ALPHA) + '/' + sub_name

                            if os.path.exists(directory):     
                                print('checking ', directory)
                                # read files
                                failfiles = {}
                                resfiles = {}
                                i = 0
                                for filename in os.listdir(directory):
                                    if filename.endswith("f.gtxt"):
                                        failfiles[i] = filename
                                    if filename.endswith("s.gtxt"):
                                        resfiles[i] = filename
                                        i+=1
                                
                                print(str(len(failfiles)) + ' fail files found')
                                print(str(len(resfiles)) + ' info files found')
                                
                                # summarize files
                                for i in range(len(failfiles)):
                                    ffile = failfiles[i]
                                    rfile = resfiles[i]
                                    
                                    initLoads = {}
                                    node_steps = []
                                    relative_loads = {}
                                    lefts = []
                                    infos = {}
                                    init_density = 0
                                    
                                    fInfo = open(directory + '/' + rfile, "r")
                                    # read initial density of graph
                                    lin = fInfo.readline()
                                    lin = lin.strip()
                                    parts = lin.split(':')
                                    init_density = float(parts[1])
                                    
                                    # read initial loads
                                    fInfo.readline() # init loads
                                    
                                    broke = False
                                    while broke==False:
                                        lin = fInfo.readline()
                                        if(lin.startswith('N')):
                                            lin = lin.strip()
                                            lin = lin[1:]
                                            parts = lin.split(':')
                                            initLoads[int(parts[0])] = float(parts[1])
                                        else:
                                            broke = True
                                    
                                    # calculate initial relative loads
                                    totalLoad = 0
                                    for node in initLoads:
                                        totalLoad += initLoads[node]
                                    for node in initLoads:
                                        relative_loads[node] = initLoads[node] * len(initLoads) / totalLoad
                                    
                                    # read results
                                    fFail = open(directory + '/' + ffile, "r")
                                    fFail.readline();   # Step 0 nothing to predict
                                    fLin = ''
                                    broke = False
                                    while broke==False:
                                        fLin = fFail.readline()
                                        if(fLin.startswith('S')):
                                            fLin = fLin.strip()
                                            fLin = fLin[1:-1]
                                            parts = fLin.split(':')
                                            lvl = int(parts[0]) - 1
                                            inParts = parts[1].split()
                                            steps = []
                                            for p in inParts:
                                                if(len(p.strip()) > 0):
                                                    np = int(p)
                                                    steps.append(np)
                                            if(len(steps) > 0):
                                                node_steps.append(steps)
                                        else:
                                            fLin = fLin.strip()
                                            fLin = fLin[5:]
                                            inParts = fLin.split()
                                            for p in inParts:
                                                if(len(p.strip()) > 0):
                                                    np = int(p)
                                                    lefts.append(np)
                                            broke = True
                                    
                                    print('step len : ' + str(len(node_steps)))
                                    step_lens.append(len(node_steps))
                                    step_step_lens.append(len(node_steps))
                                    
                                    # read step infos
                                    upBroke = False
                                    stepCnt = 0
                                    while upBroke==False:
                                        broke = False
                                        step_info = {}
                                        while broke==False:
                                            nodeCnt = 0
                                            lin = fInfo.readline()
                                            if(len(lin.strip()) == 0):
                                                broke=True
                                                upBroke=True
                                            if(lin.startswith('N')):
                                                lin = lin.strip()
                                                lin = lin[1:]
                                                parts = lin.split(':')
                                                datas = []
                                                '''
                                                Step Info per Node:
                                                    [RES]: 0=will not fail, 1=will fail
                                                    [0]  : init_load / avg_init_load
                                                    [1]  : current_load / init_load
                                                    [2]  : failed node count
                                                    [3]  : avg distance of failed nodes
                                                    [4]  : minimum distance of failed nodes
                                                    [5]  : 1st degree neighbors count - shortest distance 1
                                                    [6]  : 2nd degree neighbors count without 1st degree neighbors
                                                    [7]  : average load of 1st degree neighbors
                                                    [8]  : maximum load of 1st degree neighbors
                                                    [9]  : average load of 2nd degree neighbors
                                                    [10] : maximum load of 2nd degree neighbors
                                                    [11] : current network density
                                                    -- GENERAL GRAPH INFO --
                                                    [12] : initial graph density
                                                    [13] : node count
                                                    [14] : in sub graph connectivity probability
                                                    [15] : inter sub graph connectivity probability
                                                    [16] : gaussian mean connection
                                                    [17] : gaussian standart deviation
                                                    [18] : node tolerance percentage
                                                    [RES]: how many steps to fail, 0 if will not fail
                                                    
                                                '''
                                                node_info = int(parts[0])
                                                node_load = float(parts[1])
                                                fail_count = int(parts[2])
                                                total_fail_dist = float(parts[3])
                                                datas.append(relative_loads[node_info])
                                                datas.append(node_load / initLoads[node_info])
                                                datas.append(fail_count)
                                                datas.append(total_fail_dist / fail_count)
                                                parts = parts[4:]
                                                for part in parts:
                                                    datas.append(float(part))
                                                # add general info
                                                datas.append(init_density)
                                                datas.append(NODE_COUNT)
                                                datas.append(P_IN)
                                                datas.append(P_OUT)
                                                datas.append(GAUS_UP)
                                                datas.append(GAUS_DOWN)
                                                datas.append(ALPHA)
                                                step_info[node_info] = datas
                                            else:
                                                infos[stepCnt] = step_info
                                                stepCnt+=1
                                                broke = True
                                                           
                                    # select matched info
                                    # all failures
                                    failCnt = 0
                                    for fail_list in node_steps:
                                        for stepC in infos:
                                            if stepC <= failCnt:
                                                cnt = 0
                                                for fail in fail_list:
                                                    if(stepC in infos.keys()):
                                                        if(fail in infos[stepC].keys()):
                                                            cnt+=1
                                                            # write info line
                                                            file_indiv.write('1 ');
                                                            data = infos[stepC][fail]
                                                            for dt in data:
                                                                file_indiv.write(str(dt) + ' ')
                                                            file_indiv.write(str(1 + failCnt - stepC) + '\n')
                                        failCnt+=1
                                    
                                    # surrenders
                                    failCnt = len(node_steps) - 1
                                    fail_list = lefts
                                    for stepC in infos:
                                        if stepC <= failCnt:
                                            cnt = 0
                                            for fail in fail_list:
                                                if(stepC in infos.keys()):
                                                    if(fail in infos[stepC].keys()):
                                                        cnt+=1
                                                        # write info line
                                                        file_indiv.write('0 ');
                                                        data = infos[stepC][fail]
                                                        for dt in data:
                                                            file_indiv.write(str(dt) + ' ')
                                                        file_indiv.write('0\n')
                                                        
                        all_step_lens.append(step_step_lens.copy())
                        step_step_lens.clear()

# close files
file_indiv.close()
show_frequencies(step_lens)

'''
for stp in all_step_lens:
    if(len(stp) > 0):
        show_frequencies(stp)
'''

