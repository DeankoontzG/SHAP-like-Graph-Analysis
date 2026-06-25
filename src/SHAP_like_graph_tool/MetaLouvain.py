# -*- coding: utf-8 -*-
"""
This module implements community detection.
"""
import random
import itertools
import array

import numbers
import warnings

import networkx as nx
import numpy as np



__PASS_MAX = -1
__MIN = 0.0000001


# def count_edges_in_com_for_node(node2com,graph, node, weight):
#     com = node2com[node]
#     inc = 0.
#     for neighbor, datas in graph[node].items(): #for all neigbors of the node
#         edge_weight = datas.get(weight, 1)
#         if edge_weight <= 0:
#             error = "Bad graph type ({})".format(type(graph))
#             raise ValueError(error)
#         if node2com[neighbor] == com: #if node in looked com
#             if neighbor == node:
#                 inc += float(edge_weight)
#             else:
#                 inc += float(edge_weight) / 2.
#     return inc


def count_expected_in_com_for_node(com2nodes,graph, node, com, null_model,excluding_self=False):
    other_nodes = com2nodes[com]
    inc = 0.
    for co_com_node in other_nodes:
        if co_com_node==node and excluding_self:
            inc=inc
        else:
            inc += null_model(node, co_com_node)
    return inc

class Status(object):
    """
    To handle several data in one struct.
    Could be replaced by named tuple, but don't want to depend on python 2.6
    """

    def __init__(self):
        self.node2com = dict([])
        self.com2nodes = dict([])
        self.total_weight = 0
        self.expected = dict([])
        #self.gdegrees = dict([])
        self.internals = dict([])
        #self.loops = dict([])

    def __str__(self):
        return ("node2com : " + str(self.node2com) + " expected : "
                + str(self.expected) + " internals : " + str(self.internals)
                + " total_weight : " + str(self.total_weight))

    def copy(self):
        """Perform a deep copy of status"""
        new_status = Status()
        new_status.node2com = self.node2com.copy()
        new_status.com2nodes= self.com2nodes.copy()
        new_status.internals = self.internals.copy()
        new_status.expected = self.expected.copy()
        #new_status.gdegrees = self.gdegrees.copy()
        new_status.total_weight = self.total_weight


    """
    def init(self, graph, weight, null_model,part=None):
        #Initialize the status of a graph with every node in one community
        count = 0
        self.node2com = {}
        self.com2nodes = {}
        self.total_weight = 0
        self.expected = {}
        #self.gdegrees = dict([])
        self.internals = {}
        self.total_weight = graph.size(weight=weight)
        if part is None:
            for node in graph.nodes():
                self.node2com[node] = count
                #deg = float(graph.degree(node, weight=weight))
                #if deg < 0:
                #    error = "Bad node degree ({})".format(deg)
                #    raise ValueError(error)
                self.expected[count] = 0
                #self.gdegrees[node] = deg
                edge_data = graph.get_edge_data(node, node, default={weight: 0}) #no edge: weight at zero

                #init node-com data
                self.internals[count]=edge_data.get(weight,1) #edge with no weight: weight=1
                self.expected[count]=null_model(node,node)

                #self.loops[node] = float(edge_data.get(weight, 1))
                #self.internals[count] = self.loops[node]
                count += 1
            for node,com in self.node2com.items():
                self.com2nodes.setdefault(com,set()).add(node)
        # else:
        #     #print(part)
        #     self.node2com=part
        #     for node,com in self.node2com.items():
        #         self.com2nodes.setdefault(com,set()).add(node)
        #     #print(self.com2nodes)
        #     for node in graph.nodes():
        #         inc_edges = count_edges_in_com_for_node(self.node2com,graph,node,weight)
        #         com=self.node2com[node]
        #         inc_expected = count_expected_in_com_for_node(self.com2nodes,graph,node,com,null_model)
        #         #deg = float(graph.degree(node, weight=weight))
        #         #self.degrees[com] = self.degrees.get(com, 0) + deg
        #         #self.gdegrees[node] = deg
        #
        #         self.internals[com] = self.internals.get(com, 0) + inc_edges
        #         self.expected[com] = self.expected.get(com, 0) + inc_expected

     """
    
    def init(self, graph, weight, null_model, part=None):
        """Initialize the status of a graph with every node in one community"""
        count = 0
        self.node2com = {}
        self.com2nodes = {}
        self.total_weight = 0
        self.expected = {}
        self.internals = {}
        self.total_weight = graph.size(weight=weight)
        
        if part is None:
            # --- CAS CLASSIQUE : Initialisation en Singletons ---
            for node in graph.nodes():
                self.node2com[node] = count
                self.expected[count] = 0
                edge_data = graph.get_edge_data(node, node, default={weight: 0})

                self.internals[count] = edge_data.get(weight, 1)
                self.expected[count] = null_model(node, node)
                count += 1
            for node, com in self.node2com.items():
                self.com2nodes.setdefault(com, set()).add(node)
        else:
            # --- CAS WARM-START : Initialisation guidée décommentée et corrigée ---
            self.node2com = part.copy() # Sécurité de copie
            for node, com in self.node2com.items():
                self.com2nodes.setdefault(com, set()).add(node)
            
            # Recalcul des internes (edges) et des attendus (null model) pour chaque communauté
            for node in graph.nodes():
                com = self.node2com[node]
                
                # Calcul des arêtes internes portées par le nœud vers sa communauté
                # (On réutilise l'équivalent de count_edges_in_com_for_node de façon intégrée)
                inc_edges = 0.
                if node in graph:
                    for neighbor, datas in graph[node].items():
                        if self.node2com.get(neighbor) == com:
                            edge_weight = datas.get(weight, 1)
                            if neighbor == node:
                                inc_edges += float(edge_weight)
                            else:
                                inc_edges += float(edge_weight) / 2.
                                
                # Calcul de la masse attendue du nœud dans sa communauté selon ton null model
                inc_expected = count_expected_in_com_for_node(self.com2nodes, graph, node, com, null_model)
                
                # Agrégation dans les structures globales du statut
                self.internals[com] = self.internals.get(com, 0.) + inc_edges
                self.expected[com] = self.expected.get(com, 0.) + inc_expected

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState"
                     " instance" % seed)


def partition_at_level(dendrogram, level):

    partition = dendrogram[0].copy()
    for index in range(1, level + 1):
        for node, community in partition.items():
            partition[node] = dendrogram[index][community]
    return partition

def metamodularity(partition,graph:nx.Graph,null_model,weight="weight"):
    internal_edges=0
    internal_expected=0
    for n1,n2 in itertools.combinations_with_replacement(graph.nodes,2):
        if partition[n1]==partition[n2]:
            internal_expected+=null_model(n1,n2)
            if graph.has_edge(n1,n2):
                internal_edges +=1
    return (internal_edges-internal_expected)/graph.number_of_edges()

# def modularity(partition, graph, weight='weight'):
#
#     if graph.is_directed():
#         raise TypeError("Bad graph type, use only non directed graph")
#
#     inc = dict([])
#     deg = dict([])
#     links = graph.size(weight=weight)
#     if links == 0:
#         raise ValueError("A graph without link has an undefined modularity")
#
#     for node in graph:
#         com = partition[node]
#         deg[com] = deg.get(com, 0.) + graph.degree(node, weight=weight)
#         for neighbor, datas in graph[node].items():
#             edge_weight = datas.get(weight, 1)
#             if partition[neighbor] == com:
#                 if neighbor == node:
#                     inc[com] = inc.get(com, 0.) + float(edge_weight)
#                 else:
#                     inc[com] = inc.get(com, 0.) + float(edge_weight) / 2.
#
#     res = 0.
#     for com in set(partition.values()):
#         res += (inc.get(com, 0.) / links) - \
#                (deg.get(com, 0.) / (2. * links)) ** 2
#     return res


def best_partition(graph,
                   partition=None,
                   weight='weight',
                   resolution=1.,
                   randomize=None,
                   random_state=None,
                   null_model=None):

    dendo = generate_dendrogram(graph,
                                partition,
                                weight,
                                resolution,
                                random_state,null_model)
    return partition_at_level(dendo, len(dendo) - 1)


def generate_dendrogram(graph,
                        part_init=None,
                        weight='weight',
                        resolution=1.,
                        random_state=None,null_model=None):
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")

    random_state = check_random_state(random_state)

    # special case, when there is no link
    # the best partition is everyone in its community
    if graph.number_of_edges() == 0:
        part = dict([])
        for i, node in enumerate(graph.nodes()):
            part[node] = i
        return [part]

    current_graph = graph.copy()
    status = Status()
    status.init(current_graph, weight, null_model,part_init)
    status_list = list()

    #print("----will do first level---")
    __one_level(current_graph, status, weight, resolution, random_state, null_model)

    new_mod = __modularity(status, resolution)
    partition = __renumber(status.node2com)
    status_list.append(partition)
    mod = new_mod
    current_graph = induced_graph(partition, current_graph, weight)
    current_null_model=induced_null_model(partition,null_model)
    #print(current_graph.nodes())
    #print(current_null_model(0,1))
    #print(partition)
    status.init(current_graph, weight,current_null_model)

    while True:
        #print("----will do new level---")

        __one_level(current_graph, status, weight, resolution, random_state,current_null_model)
        new_mod = __modularity(status, resolution)
        if new_mod - mod < __MIN:
            break
        partition = __renumber(status.node2com)
        status_list.append(partition)
        mod = new_mod
        current_graph = induced_graph(partition, current_graph, weight)
        current_null_model = induced_null_model(partition, current_null_model)
        status.init(current_graph, weight,current_null_model)
    return status_list[:]


def induced_graph(new_node2com, graph, weight="weight"):

    ret = nx.Graph()
    ret.add_nodes_from(new_node2com.values())

    for node1, node2, datas in graph.edges(data=True):
        edge_weight = datas.get(weight, 1)
        com1 = new_node2com[node1]
        com2 = new_node2com[node2]
        w_prec = ret.get_edge_data(com1, com2, {weight: 0}).get(weight, 1)
        ret.add_edge(com1, com2, **{weight: w_prec + edge_weight})

    return ret

def induced_null_model(new_node2com, null_model):

    new_null_model={}
    #ret = nx.Graph()
    #ret.add_nodes_from(new_node2com.values())

    new_com2nodes={}
    for node, com in new_node2com.items():
        new_com2nodes.setdefault(com, set()).add(node)
    #print("new level coms: "+str(len(new_com2nodes)))

    for com1,com2 in itertools.combinations_with_replacement(new_com2nodes.keys(),2):
        len_com1=len(new_com2nodes[com1])
        len_com2=len(new_com2nodes[com2])
        nb_combinations=len_com1*len_com2/2
        if nb_combinations>100000:
            requested_size = max([len_com1, len_com2]) * 20
            sample_size = min([len_com1, len_com2, requested_size])
            
            sample=zip(random.sample(new_com2nodes[com1],sample_size),random.sample(new_com2nodes[com2],sample_size))
        else:
            if com1==com2:
                sample=itertools.combinations_with_replacement(new_com2nodes[com1],2)
            else:
                sample=itertools.product(new_com2nodes[com1],new_com2nodes[com2])

        new_null_model[frozenset([com1,com2])]=sum([null_model(u,v) for u,v in sample])

    update_null_model= lambda u,v: new_null_model[frozenset([u,v])]
    #print(new_null_model)
    return update_null_model


def __renumber(dictionary):
    """Renumber the values of the dictionary from 0 to n
    """
    values = set(dictionary.values())
    target = set(range(len(values)))

    if values == target:
        # no renumbering necessary
        ret = dictionary.copy()
    else:
        # add the values that won't be renumbered
        renumbering = dict(zip(target.intersection(values),
                               target.intersection(values)))
        # add the values that will be renumbered
        renumbering.update(dict(zip(values.difference(target),
                                    target.difference(values))))
        ret = {k: renumbering[v] for k, v in dictionary.items()}

    return ret

def __one_level(graph, status, weight_key, resolution, random_state,null_model):
    """Compute one level of communities
    """
    modified = True
    nb_pass_done = 0
    cur_mod = __modularity(status, resolution)
    new_mod = cur_mod

    while modified and nb_pass_done != __PASS_MAX:
        #print("----- doing one pass over all nodes")
        #print("modul",__modularity(status, resolution))
        cur_mod = new_mod
        modified = False
        nb_pass_done += 1

        for node in __randomize(graph.nodes(), random_state):
            com_node = status.node2com[node]
            #degc_totw = status.gdegrees.get(node, 0.) / (status.total_weight * 2.)  # NOQA
            neigh_communities = __neighcom(node, graph, status, weight_key) #dict(com,weight)
            internal_edges_removed= resolution * neigh_communities.get(com_node,0)
            #expected_internal_removed = (status.degrees.get(com_node, 0.) - status.gdegrees.get(node, 0.)) * degc_totw
            expected_internal_removed = count_expected_in_com_for_node(status.com2nodes,graph,node,com_node,null_model,excluding_self=False)
            remove_cost = - internal_edges_removed + expected_internal_removed

            __remove(node, com_node,
                     neigh_communities.get(com_node, 0.), expected_internal_removed,status)
            best_com = com_node
            best_increase = -100000
            best_com_expected_added=0
            if node == "FR":

                print("--- com: ",com_node,status.com2nodes[com_node])
            for com, dnc in __randomize(neigh_communities.items(), random_state):

                #incr = remove_cost + resolution * dnc - status.degrees.get(com, 0.) * degc_totw
                #expected_added = status.degrees.get(com, 0.) * degc_totw
                expected_added = count_expected_in_com_for_node(status.com2nodes,graph,node,com,null_model,excluding_self=True)+null_model(node,node)
                incr = remove_cost + resolution * dnc - expected_added
                if node == "FR":
                    # print("---")
                    print("--gain", node,com, dnc, status.com2nodes[com], incr,dnc - expected_added)
                if incr > best_increase: ### HERE maybe allow to go back to singleton
                    best_increase = incr
                    best_com = com
                    best_com_expected_added=expected_added
                    to_write_gain=dnc

            if best_com != com_node:
                modified = True

            #print("remove", node, best_com, - resolution * neigh_communities.get(com_node,0),expected_internal_removed)
            if node=="FR":
                print("=>", node,"from",com_node,"to",best_com,to_write_gain,status.com2nodes[best_com], best_increase,status.expected[best_com])

            __insert(node, best_com,
                     neigh_communities.get(best_com, 0.), best_com_expected_added,status)



        new_mod = __modularity(status, resolution)
        if new_mod - cur_mod < __MIN:
            break


def __neighcom(node, graph, status, weight_key):
    """
    Compute the communities in the neighborhood of node in the graph given
    with the decomposition node2com
    """
    weights = {}
    for neighbor, datas in graph[node].items():
        #if neighbor != node:
        edge_weight = datas.get(weight_key, 1)
        neighborcom = status.node2com[neighbor]
        weights[neighborcom] = weights.get(neighborcom, 0) + edge_weight

    return weights


def __remove(node, com, edges_to_other_nodes, expect_to_other_nodes,status):
    """ Remove node from community com and modify status"""
    #status.degrees[com] = (status.degrees.get(com, 0.)
    #                       - status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) -
                                  edges_to_other_nodes)# - status.loops.get(node, 0.))
    status.expected[com] -= expect_to_other_nodes
    #if status.expected[com]<-0.001:
        #print("WARNING",node,com,status.expected[com],expect_to_other_nodes,status.com2nodes[com])
    status.node2com[node] = -1
    status.com2nodes[com].remove(node)


def __insert(node, com, edges_to_other_nodes, expect_to_other_nodes,status):
    """ Insert node into community and modify status"""
    status.node2com[node] = com
    status.com2nodes[com].add(node)
    #status.degrees[com] = (status.degrees.get(com, 0.) +
    #                       status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) +
                                  edges_to_other_nodes)# + status.loops.get(node, 0.))
    status.expected[com] += expect_to_other_nodes
    #if status.expected[com]<=-0.001:
 #       print("weird insert",status.expected[com],expect_to_other_nodes,node,status.com2nodes[com])


def __modularity(status, resolution):
    """
    Fast compute the modularity of the partition of the graph using
    status precomputed
    """
    links = float(status.total_weight)
    result = 0.
    for community in set(status.node2com.values()):
        in_edges = status.internals.get(community, 0.)
        #degree = status.degrees.get(community, 0.)
        observed_edges=in_edges * resolution / links
        #expected_edges=((degree / (2. * links)) ** 2)
        expected_edges=status.expected.get(community, 0.) /links
        if links > 0:
            result += observed_edges -  expected_edges
        # print("---modularity: com:",community,"edges:",observed_edges*links,"expected: ",expected_edges*links,status.com2nodes[community])
    return result

    


def __randomize(items, random_state):
    """Returns a List containing a random permutation of items"""
    randomized_items = list(items)
    random_state.shuffle(randomized_items)
    return randomized_items