# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 12:27:34 2019

@author: Ahmert
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
import sys
import time
from collections import Counter
import numpy as np
from random import choice

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
    print('Calculating Graph Stats')
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
    
    print("density: %s" % nx.density(G))
    
    dist.clear()
    for p in pathlengths:
        if p in dist:
            dist[p] += 1
        else:
            dist[p] = 1
    
    print('')
    if(len(dist) > 0):
        print("length #paths")
        verts = dist.keys()
        for d in sorted(verts):
            print('%s %d' % (d, dist[d]))

# to find the node with minimum load
def find_min_shrt_path_node(initial):
    minV = 100000
    nodeV = -1
    for i in range(len(loads)):
        if(existing[i]):
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
def find_max_eff_node():
    maxV = 0
    nodeV = -1
    for i in range(len(efficiencies)):
        if(existing[i]):
            if(efficiencies[i] > maxV):
                maxV = efficiencies[i]
                nodeV = loadids[i]
    
    return nodeV
    
# to get the node to be attacked according to compare mode
def get_strongest_node(initial):
    if COMPARE_MODE == MODE_EFFICIENCY:
        return find_max_eff_node()
    elif COMPARE_MODE == MODE_LOAD:
        return find_min_shrt_path_node(initial)
    return -2;
    

# to delete a node without removing from the graph
def delete_node(num):
    #print('deleting node edges: ', num)
    existing[num] = False
    edgesN = []
    for ed in G.edges(num):
        edgesN.append(ed)
    #print(edgesN)
    G.remove_edges_from(edgesN)

# to define node colors according to load or efficiency
def recolor():
    print('recoloring')
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
    for i in range(len(existing)):
        if(existing[i]):
            if COMPARE_MODE == MODE_LOAD :
                if(loads[i] > initial_loads[i] * (1 + alphas[ALPHA])):
                    rmd += 1
                    delete_node(i)
            elif COMPARE_MODE == MODE_EFFICIENCY:
                if(efficiencies[i] * (1 + alphas[ALPHA]) < initial_efficiencies[i]):
                    rmd += 1
                    delete_node(i)
    
    if rmd > 0:
        print(rmd, ' nodes are removed!')
        return True
        
    return False
    
# to calculate the efficiencies
def find_graph_efficiency(mode_eff):
    print('Calculating Efficiencies')
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
        calculate_graph_stats()
        find_graph_efficiency(1)
        if(remove_failures()):
            print('failures removed')
            turns[0] = turns[0] + 1
            turns[1] = turns[1] + 1
            calculate_graph_stats()
            find_graph_efficiency(1)
            recolor()
            redraw()              
    elif event.key == 'i':
        insert_guard(30, 3)
    elif event.key == 'b':
        print('s_f')
        show_frequencies()
            
# auto attack on network
def auto_attack(num, initial):
    for n in range(num):
        calculate_graph_stats()
        find_graph_efficiency(1)
        minSPNode = get_strongest_node(initial)
        if (minSPNode >= 0):
            print('removing node: ', minSPNode)
            turns[0] = turns[0] + 1
            delete_node(minSPNode)
            calculate_graph_stats()
            find_graph_efficiency(1)
            recolor()
            
        if(VIS_ON):
            fig.canvas.draw()
            # nx.draw(G, with_labels=True)
            redraw()
            
# auto fail on network
def auto_failure(num):
    for n in range(num):
        alive_ids = []
        alive_ids.clear()
        for idn in range (len(existing)):
            if(existing[idn]):
                alive_ids.append(idn)
        if(len(alive_ids) > 0):
            selected = random.randint(0, len(alive_ids)-1)
            print('removing node: ', alive_ids[selected])
            turns[0] = turns[0] + 1
            delete_node(alive_ids[selected])
            calculate_graph_stats()
            find_graph_efficiency(1)
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
            
# auto attack on network
def auto_cascade():
    isOn = True
    while isOn:
        calculate_graph_stats()
        find_graph_efficiency(1)
        isOn = False
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
    
    indexes = np.arange(len(labels))
    width = 1
    
    print(labels)
    print(values)
    
    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels)
    plt.show()

# to delete isolated nodes
def remove_isolates():
    # remove isolated nodes that have no neighbors
    isols = nx.isolates(G)
    for iso in isols:
        delete_node(iso)
        
# to insert guard to network
def insert_guard(cover, strength):
    ''' 
    print(len(G.nodes()))
    a = len(G.nodes())
    n = random.randint(0,a)
    print('>', n)
    '''
    selecteds = []
    sel_cnt = 0
    while sel_cnt < cover:
        lena = len(G.nodes())
        n = random.randint(0, lena)
        # print('>', n)
        if n in selecteds:
            continue
        sel_cnt += 1
        selecteds.append(n)
    maxum = len(G.nodes()) + 1

    print('---')
    to_add = 0
    while to_add < strength:
        G.add_node(maxum)
        sel_cnt = 0
        while sel_cnt < cover:
            G.add_edge(maxum, selecteds[sel_cnt])
            sel_cnt += 1
        to_add += 1
        maxum += 1
    print('------')
    return maxum



# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# ********************************** defines **********************************
NODE_COUNT = 100
P_IN = 0.2
P_OUT = 0.04
GAUS_MEAN = 20
GAUS_ST_DEV = 10
ALPHA = 4
ATTACK_COUNT = 5
VIS_ON = True # show plot
MODE_EFFICIENCY = 2;
MODE_LOAD = 1;
COMPARE_MODE = MODE_EFFICIENCY
MODE_ATTACK = 11;
MODE_ATTACK_DYNAMIC = 12;
MODE_FAILURE = 13;
DROP_MODE = MODE_ATTACK

# ****************************** generate graph *******************************
G = my_gaussian_random_partition_graph(NODE_COUNT, GAUS_MEAN, GAUS_ST_DEV, P_IN, P_OUT)

show_frequencies()

print("density: %s" % nx.density(G))

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

# ******************************* pre processes *******************************
# initialize
existing.clear()
for node in G:
    existing.append(True)
    color_map.append('blue')

turns = [0, 0]
alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.33, 0.5, 0.75, 1.0]

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

# visualize
if(VIS_ON):
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title('Step 0')
    nx.draw(G, with_labels=True, ax=ax)
    fig.canvas.mpl_connect('key_press_event', key_press)
    plt.show()

# auto process
# auto_drop(ATTACK_COUNT)
#auto_cascade()
'''
if(VIS_ON):
    calculate_graph_stats()
    recolor()
    redraw()
'''