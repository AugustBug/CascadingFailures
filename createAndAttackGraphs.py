# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 02:24:48 2019

@author: Ahmert
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
import sys
import os
import time
from collections import Counter
import numpy as np

# to get current time ms
current_milli_time = lambda: int(round(time.time() * 1000))

# to generate a gaussian graph. default version in lib has errors
def my_gaussian_random_partition_graph(n, s, v, p_in, p_out):
    if s > n:
        raise nx.NetworkXError("s must be <= n")
    assigned = 0
    sizes = []
    while True:
        #size = int(None.gauss(s, float(s) / v + 0.5))
        size = int(random.normalvariate(s, v))
        if size < 1:  # how to handle 0 or negative sizes?
            continue
        if assigned + size >= n:
            sizes.append(n-assigned)
            break
        assigned += size
        sizes.append(size)
    return nx.random_partition_graph(sizes, p_in, p_out, False, None)

# calculate graph properties for nodes
def calculate_graph_stats():
    #print('Calculating Graph Stats')
    pathlengths.clear()
    loads.clear()
    loadids.clear()
    for v in G.nodes():
        if existing[v]:
            spl = dict(nx.single_source_shortest_path_length(G, v))
            # print('*', v, " = ", end='')
            top = 0
            cnt = 0
            for p in spl:
                top += spl[p]
                cnt += 1
                pathlengths.append(spl[p])
            avgSP = top/cnt
            # print(top, " > ", avgSP)
            loads.append(10000 if avgSP == 0 else avgSP)
            loadids.append(v)
        else:
            loads.append(20000)
            loadids.append(v)
    
    print('')
    if(len(pathlengths) > 0):
        print("average load %s" % (sum(pathlengths) / len(pathlengths)))
        avgLoad[0] = sum(pathlengths) / len(pathlengths)
        print("minimum load %s" % (min(loads)))
    
    dist.clear()
    for p in pathlengths:
        if p in dist:
            dist[p] += 1
        else:
            dist[p] = 1
    '''
    print('')
    if(len(dist) > 0):
        print("length #paths")
        verts = dist.keys()
        for d in sorted(verts):
            print('%s %d' % (d, dist[d])) '''

# to find the node with minimum load
def find_min_shrt_path_node(initial, first):
    minV = 100000
    nodeV = -1
    for i in range(len(loads)):
        if(existing[i]):
            if(is_in_target(i) or first):
                if(initial):
                    if(initial_loads[i] < minV):
                        minV = initial_loads[i]
                        nodeV = loadids[i]
                else:
                    if(loads[i] < minV):
                        minV = loads[i]
                        nodeV = loadids[i]
    
    return nodeV

# to find the node with maximum efficiency
def find_max_eff_node(first):
    maxV = 0
    nodeV = -1
    for i in range(len(efficiencies)):
        if(existing[i]):
            if(is_in_target(i) or first):
                if(efficiencies[i] > maxV):
                    maxV = efficiencies[i]
                    nodeV = loadids[i]
    
    return nodeV
        
# to get the node to be attacked according to compare mode
def get_strongest_node(initial, first):
    if COMPARE_MODE == MODE_EFFICIENCY:
        return find_max_eff_node(first)
    elif COMPARE_MODE == MODE_LOAD:
        return find_min_shrt_path_node(initial, first)
    return -2;
    

# to delete a node without removing from the graph
def delete_node(num, is_fail):
    # print('deleting node edges: ', num)
    # if node is deleted in process, record the fail data
    if is_fail == True:
        fail_info_update(num)
        
    existing[num] = False
    edgesN = []
    for ed in G.edges(num):
        edgesN.append(ed)
    # print(edgesN)
    G.remove_edges_from(edgesN)

# to define node colors according to load or efficiency
def recolor():
    #print('recoloring')
    for i in range(len(existing)):
        if(existing[i]):
            if COMPARE_MODE == MODE_LOAD :
                if(loads[i] < initial_loads[i]):
                    color_map[i] = 'green'
                elif(loads[i] < initial_loads[i] * (1 + alphas[ALPHA] / 2)):
                    color_map[i] = 'blue'
                elif(loads[i] < initial_loads[i] * (1 + alphas[ALPHA])):
                    color_map[i] = 'yellow'
                else:
                    color_map[i] = 'red'
            elif COMPARE_MODE == MODE_EFFICIENCY:
                if(efficiencies[i] > initial_efficiencies[i]):
                    color_map[i] = 'green'
                elif(efficiencies[i] * (1 + alphas[ALPHA] / 2) > initial_efficiencies[i]):
                    color_map[i] = 'blue'
                elif(efficiencies[i] * (1 + alphas[ALPHA]) > initial_efficiencies[i]):
                    color_map[i] = 'yellow'
                else:
                    color_map[i] = 'red'
        else:
            color_map[i] = 'white'
            
# to delete the nodes that are non functional
def remove_failures():
    rmd = 0
    fileFails.write('S' + str(turns[1]) + ':')
    for i in range(len(existing)):
        if(existing[i]):
            if COMPARE_MODE == MODE_LOAD :
                if(loads[i] > initial_loads[i] * (1 + alphas[ALPHA])):
                    rmd += 1
                    delete_node(i, True)
                    fileFails.write(str(i) + ' ')
            elif COMPARE_MODE == MODE_EFFICIENCY:
                if(efficiencies[i] * (1 + alphas[ALPHA]) < initial_efficiencies[i]):
                    rmd += 1
                    delete_node(i, True)
                    fileFails.write(str(i) + ' ')
    
    fileFails.write(';\n')
    if rmd > 0:
        print(rmd, ' failures!')
        return True
        
    return False
    
# to calculate the efficiencies
def find_graph_efficiency(mode_eff):
    #print('Calculating Efficiencies')
    total_node_count = 0
    efficiencies.clear()
    for v in G.nodes():
        if existing[v]:
            spls = dict(nx.single_source_shortest_path_length(G, v))
            temp = []
            for k in spls:
                temp.append(spls[k])
            spl_freqs = Counter(temp)
            single_total_eff = 0;
            nums, values = zip(*spl_freqs.items())
            for i in range(len(nums)):
                if(nums[i] != 0):
                    single_total_eff += values[i]/nums[i]
            efficiencies.append(single_total_eff)
            total_node_count += 1
        else:
            efficiencies.append(-1);

    total_eff = 0        
    for ef in efficiencies:
        if(ef >= 0):
            total_eff += ef
    if(total_node_count > 0):
        if(mode_eff == 0):
            total_efficiencies[0] = total_eff / total_node_count
        else:
            total_efficiencies[1] = total_eff / total_node_count
            print('efficiency: ', total_efficiencies[1] / total_efficiencies[0])
    else:
        print('efficiency: 0')
    
# to draw the graph according to states
def redraw():
    plt.cla()
    nx.draw_circular(G, with_labels=True, ax=ax, node_color = color_map)
    fig.canvas.draw()
    print('turn ', turns[1], "<=>", turns[0], ".")
    ax.set_title('Step {}'.format(turns[1]))
    fig.canvas.draw()

# key press binding on plot
def key_press(event):
    sys.stdout.flush()
    if event.key == ' ':
        auto_drop(1)
    elif event.key == 'c':
        redraw()
        recolor()
    elif event.key == 'm':
        calculate_process()
        write_stats()
        if(remove_failures()):
            print('failures removed')
            turns[0] = turns[0] + 1
            turns[1] = turns[1] + 1
            calculate_process()   
            recolor()
            redraw()
            
# to write the info to the files
def write_stats():
    '''
    Step Info per Node:
        [0]  : node id
        [1]  : current load
        [2]  : failed node count
        [3]  : total distance of failed nodes
        [4]  : minimum distance of failed nodes
        [5]  : 1st degree neighbors count - shortest distance 1
        [6]  : 2nd degree neighbors count without 1st degree neighbors
        [7]  : average load of 1st degree neighbors
        [8]  : maximum load of 1st degree neighbors
        [9]  : average load of 2nd degree neighbors
        [10] : maximum load of 2nd degree neighbors
        [11] : current network density
    '''
    fileSteps.write('S' + str(turns[1]) + ':\n')
    for v in G.nodes():
        if existing[v]:
            if ((COMPARE_MODE == MODE_LOAD) and (loads[v] < 100)) or \
                ((COMPARE_MODE == MODE_EFFICIENCY) and (efficiencies[v] >= 0)):
                # direct neigbors
                negs = list(G.neighbors(v))
                negCnt = len(negs)
                maxNegLoad = 0
                totNegLoad = 0
                for nnn in negs:
                    relVal = 0
                    if COMPARE_MODE == MODE_LOAD: 
                        relVal = loads[nnn] / initial_loads[nnn]
                    elif COMPARE_MODE == MODE_EFFICIENCY: 
                        relVal = efficiencies[nnn] / initial_efficiencies[nnn]

                    if relVal > maxNegLoad:
                        maxNegLoad = relVal
                    totNegLoad += relVal
                    
                # 2nd degree neigbors
                lis2neg = []
                for vvv in negs:
                    lis2neg = lis2neg + list(set(list(G.neighbors(vvv))) - set(lis2neg))
                
                lis2neg = lis2neg + list(set(negs) - set(lis2neg))
                
                neg2Cnt = len(lis2neg)
                maxNeg2Load = 0
                totNeg2Load = 0
                for nnn in lis2neg:
                    relVal = 0
                    if COMPARE_MODE == MODE_LOAD: 
                        relVal = loads[nnn] / initial_loads[nnn]
                    elif COMPARE_MODE == MODE_EFFICIENCY: 
                        relVal = efficiencies[nnn] / initial_efficiencies[nnn]

                    if relVal > maxNeg2Load:
                        maxNeg2Load = relVal
                    totNeg2Load += relVal
                
                base_val = loads[v]
                if COMPARE_MODE == MODE_EFFICIENCY: 
                    base_val = efficiencies[v]

                fileSteps.write('N' + str(v) + ':' + str(base_val) + ':' + str(fail_count[0]) + \
                                ':' + str(fail_total_distances[v]) + ':' + str(fail_min_distances[v]) + \
                                ':' + str(negCnt) + ':' + str(neg2Cnt) + \
                                ':' + str(0 if negCnt<1 else totNegLoad/negCnt) + ':' + str(maxNegLoad) + \
                                ':' + str(0 if neg2Cnt<1 else totNeg2Load/neg2Cnt) + ':' + str(maxNeg2Load) + \
                                ':' + str(nx.density(G)) + '\n')

# auto attack on network
def auto_attack(num, initial):
    for n in range(num):
        print('attack ', n)
        calculate_process()
        minSPNode = get_strongest_node(initial, n==0)
        if (minSPNode >= 0):
            #print('removing node: ', minSPNode)
            # add neighbors to targets list
            negs = G.neighbors(minSPNode)
            for neg in negs:
                add_to_targets(neg)
            
            turns[0] = turns[0] + 1
            delete_node(minSPNode, True)
            # calculate_process()
            if(VIS_ON):
                recolor()
            
        if(VIS_ON):
            fig.canvas.draw()
            # nx.draw(G, with_labels=True)
            redraw()
            
# auto fail on network
def auto_failure(num):
    for n in range(num):
        print('random failure ', n)
        alive_ids = []
        alive_ids.clear()
        for idn in range (len(existing)):
            if(existing[idn]):
                alive_ids.append(idn)
        if(len(alive_ids) > 0):
            selected = random.randint(0, len(alive_ids)-1)
            #print('removing node: ', alive_ids[selected])
            turns[0] = turns[0] + 1
            delete_node(alive_ids[selected], True)
            calculate_process()
            if(VIS_ON):
                recolor()
            
        if(VIS_ON):
            fig.canvas.draw()
            # nx.draw(G, with_labels=True)
            redraw()
        
# auto drop nodes    
def auto_drop(num):
    if(DROP_MODE == MODE_ATTACK_DYNAMIC):
        auto_attack(num, False)
    elif(DROP_MODE == MODE_FAILURE):
        auto_failure(num)
    elif(DROP_MODE == MODE_ATTACK):
        auto_attack(num, True)
    elif(DROP_MODE == MODE_ATTACK_GROUP):
        auto_attack(num, False)     # uses dynamic attack, but activates dynamic list for attack targets

# auto attack on network
def auto_cascade():
    isOn = True
    while isOn:
        calculate_process()
        isOn = False
        write_stats()
        if(remove_failures()):
            turns[0] = turns[0] + 1
            turns[1] = turns[1] + 1
            print('Auto Turn ' + str(turns[1]))
            isOn = True
    
# to show node degree frequencies chart
def show_frequencies():
    degrees = [val for (node, val) in G.degree()]
    degrees.sort()
    freqs = Counter(degrees)
    
    labels, values = zip(*freqs.items())
    
    print(labels)
    print(values)
    
# to delete isolated nodes
def remove_isolates():
    # remove isolated nodes that have no neighbors
    isols = nx.isolates(G)
    for iso in isols:
        delete_node(iso, False)

def fail_info_update(node):
    fail_count[0] += 1

    for v in G.nodes():
        if existing[v]:
            if node != v:
                try:
                    p=nx.shortest_path_length(G, source=v, target=node)
                    if p>0:
                        fail_total_distances[v] += p
                        if p < fail_min_distances[v]:
                            fail_min_distances[v] = p
                except nx.NetworkXNoPath:
                    p=-1

def calculate_process():
    print("density: %s" % nx.density(G))
    if COMPARE_MODE == MODE_LOAD:
        calculate_graph_stats()
    elif COMPARE_MODE == MODE_EFFICIENCY:
        find_graph_efficiency(1)
            
def is_in_target(id):
    if DROP_MODE == MODE_ATTACK_GROUP:
        for el in target_sub:
            if el == id:
                return True
    else:
        return True
    return False

def add_to_targets(id):
    for el in target_sub:
        if el == id:
            return False
    target_sub.append(id)
        

    
    
    
        
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# ********************************** defines **********************************
NODE_COUNT = 500
P_IN = 0.075
P_OUT = 0.016
GAUS_MEAN = 30
GAUS_ST_DEV = 18
ALPHA = 7
ATTACK_COUNT = 113
VIS_ON = False # show plot
MODE_EFFICIENCY = 2;
MODE_LOAD = 1;
COMPARE_MODE = MODE_EFFICIENCY
MODE_ATTACK = 11;
MODE_ATTACK_DYNAMIC = 12;
MODE_FAILURE = 13;
MODE_ATTACK_GROUP = 14;
DROP_MODE = MODE_ATTACK_GROUP

alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.33, 0.5, 0.75, 1.0]
        
tr_num = 0
while True:
# ****************************** generate graph *******************************
    print('----------', tr_num)
    G = my_gaussian_random_partition_graph(NODE_COUNT, GAUS_MEAN, GAUS_ST_DEV, P_IN, P_OUT)
    print("initial density: %s" % nx.density(G))
    print('edges: ', G.number_of_edges())
    show_frequencies()
    
# ********************************* variables *********************************
    dist = {}
    pathlengths = []
    initial_loads = []
    loads = []
    loadids = []
    avgLoad = [0]
    initAvg = [0]
    total_efficiencies = [0, 0]
    initial_efficiencies = []
    efficiencies = []
    existing = []
    color_map = []
    fail_count = [0]
    fail_total_distances = []
    fail_min_distances = []
    target_sub = []
    
# ******************************* pre processes *******************************
    # initialize
    existing.clear()
    for node in G:
        existing.append(True)
        color_map.append('blue')
        fail_total_distances.append(0)
        fail_min_distances.append(NODE_COUNT)
    
    turns = [0, 0]
    
    # remove isolated nodes
    remove_isolates()
    
    # initial stats
    calculate_graph_stats()
    
    # set initial loads
    initAvg[0] = avgLoad[0]
    for ll in loads:
        initial_loads.append(ll)
    
    # initial efficiencies
    find_graph_efficiency(0)
    # set initial efficiencies
    for ff in efficiencies:
        initial_efficiencies.append(ff)
        
    target_sub.clear()
    
# ****************************** file initialize ******************************
    # file process
    sub_name = 'LOAD'
    if(COMPARE_MODE == MODE_EFFICIENCY):
        sub_name = 'EFF'
    if(DROP_MODE == MODE_ATTACK_DYNAMIC):
        sub_name = sub_name + '_DYNATT'
    elif(DROP_MODE == MODE_ATTACK):
        sub_name = sub_name + '_ATTACK'
    elif(DROP_MODE == MODE_FAILURE):
        sub_name = sub_name + '_FAILURE'
    elif(DROP_MODE == MODE_ATTACK_GROUP):
        sub_name = sub_name + '_GRPATT'
        
    directory = './GR_TEST/NODE_' + str(NODE_COUNT) + '/DENS_' + str(GAUS_MEAN)\
    + '_' + str(GAUS_ST_DEV) + '_' + str(P_IN) + '_' + str(P_OUT) + '/ALPHA_'\
    + str(alphas[ALPHA]) + '/' + sub_name
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    curtim = current_milli_time()
    fileFails = open(directory + '/' + str(curtim) + '_f.gtxt', "w")
    fileSteps = open(directory + '/' + str(curtim) + '_s.gtxt', "w")
    
# **************************** file write initials ****************************
    fileSteps.write('INIT DENSITY:'+ str(nx.density(G)) + '\n')
    if COMPARE_MODE == MODE_LOAD:
        fileSteps.write('INIT LOADS:\n')
        for v in G.nodes():
            if initial_loads[v] < 100:
                fileSteps.write('N' + str(v) + ':' + str(initial_loads[v]) + '\n')
    elif COMPARE_MODE == MODE_EFFICIENCY:
        fileSteps.write('INIT EFFICIENCIES:\n')
        for v in G.nodes():
            if initial_efficiencies[v] >= 0:
                fileSteps.write('N' + str(v) + ':' + str(initial_efficiencies[v]) + '\n')
# ********************************* visualize *********************************
    if(VIS_ON):
        # visualize
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_title('Step 0')
        nx.draw(G, with_labels=True, ax=ax)
        fig.canvas.mpl_connect('key_press_event', key_press)
        plt.show()
    
    
# ********************************* operation *********************************
    # auto process
    auto_drop(ATTACK_COUNT)
    auto_cascade()
    
    print('cascading finished!')
# ****************************** file write tail ******************************
    fileFails.write('LEFT:')
    for v in G.nodes():
        if existing[v]:
            fileFails.write(str(v) + ' ')
    fileFails.write('\n')
    
# ********************************** results **********************************
    if(VIS_ON):
        calculate_process()
        recolor()
        redraw()
    
    # close files
    fileFails.close()
    fileSteps.close()
