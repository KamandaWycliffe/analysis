#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from itertools import permutations
import networkx as nx
import math
import random


# In[7]:


'''from functions.matching import find_minimum_weight_matching leads to a mapping problem in python 3
We will therefore use a different minimum weight matching approach with almost similar results'''

#from functions.matching import find_minimum_weight_matching

def solve_tsp(points):
    # build a graph
    Graph = build_graph(points)
    graph = Graph

    # Define the minimum spanning tree
    minimum_spanning_tree = minimum_spanning_tree_fun(Graph)
    #print("minimum_spanning_tree: ", minimum_spanning_tree)

    # get the odd vertices
    oddvertices = find_oddvertices(minimum_spanning_tree)
    #print("Odd vertices in minimum_spanning_tree: ", oddvertices)

    # Append minimum weight matching edges to minimum_spanning_tree
    min_weight = minimum_weight_matching(minimum_spanning_tree, Graph, oddvertices)
    #print("Minimum weight matching: ", minimum_spanning_tree)

    # find a euler tour (e_tour)
    e_tour = find_e_tour(minimum_spanning_tree, Graph)
    tour = e_tour


    cost = cost_of_tour(tour, graph)

    print ("\nPoints: ",points)
    print("\n Travelling Sales Man tour: ", e_tour)
    print("\nCost of the salesman tour: ", cost)

    return cost, tour,graph, minimum_spanning_tree

#Get euclidean distance between two points
def get_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def build_graph(points):
    graph = {}
    for current in range(len(points)):
        for next_point in range(len(points)):
            if current != next_point:
                if current not in graph:
                    graph[current] = {}

                graph[current][next_point] = get_distance(points[current][0], points[current][1], points[next_point][0],
                                                        points[next_point][1])

    return graph


class jonint_find:
    def __init__(self):
        self.weights = {}
        self.parents = {}

    def __getitem__(self, object):
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # find sales tour of objects leading to the source
        sales_tour = [object]
        source = self.parents[object]
        while source != sales_tour[-1]:
            sales_tour.append(source)
            source = self.parents[source]

        # compress the sales_tour and return
        for ancestor in sales_tour:
            self.parents[ancestor] = source
        return source

    def __iter__(self):
        return iter(self.parents)

    def union(self, *objects):
        sources = [self[x] for x in objects]
        most_weighted = max([(self.weights[r], r) for r in sources])[1]
        for r in sources:
            if r != most_weighted:
                self.weights[most_weighted] += self.weights[r]
                self.parents[r] = most_weighted


def minimum_spanning_tree_fun(Graph):
    tree = []
    branches = jonint_find()
    for W, u, v in sorted((Graph[u][v], u, v) for u in Graph for v in Graph[u]):
        if branches[u] != branches[v]:
            tree.append((u, v, W))
            branches.union(u, v)

    return tree
#get cost of the salesman tour
def cost_of_tour(tour, graph):
    """Returns Cost of the salesman tour"""
    cost_of_travel = 0
    tour = tour[0:len(tour)-1]
    preceding_vertex = tour[len(tour)-1]
    for present_vertex in tour:
        if preceding_vertex > present_vertex:
            cost_of_travel = cost_of_travel + graph[present_vertex][preceding_vertex]
        else:
            cost_of_travel = cost_of_travel + graph[preceding_vertex][present_vertex]
        preceding_vertex = present_vertex
    return cost_of_travel



def minimum_weight_matching(minimum_spanning_tree, Graph, oddvertices):
    
    random.shuffle(oddvertices)

    while oddvertices:
        v = oddvertices.pop()
        distance = float("inf")
        u = 1
        closest = 0
        for u in oddvertices:
            if v != u and Graph[v][u] < distance:
                distance = Graph[v][u]
                closest = u

        minimum_spanning_tree.append((v, closest, distance))
        oddvertices.remove(closest)


def find_oddvertices(minimum_spanning_tree):
    temp_gr = {}
    vertices = []
    for ej in minimum_spanning_tree:
        if ej[0] not in temp_gr:
            temp_gr[ej[0]] = 0

        if ej[1] not in temp_gr:
            temp_gr[ej[1]] = 0

        temp_gr[ej[0]] += 1
        temp_gr[ej[1]] += 1

    for vertex in temp_gr:
        if temp_gr[vertex] % 2 == 1:
            vertices.append(vertex)

    return vertices
        
        
def find_e_tour(matched_minimum_spanning_tree, Graph):
    # get ne1igbours
    n = {}
    for ej in matched_minimum_spanning_tree:
        if ej[0] not in n:
            n[ej[0]] = []

        if ej[1] not in n:
            n[ej[1]] = []

        n[ej[0]].append(ej[1])
        n[ej[1]].append(ej[0])

    print("Neighbours: ", n)

    # obtains the hamiltonian circuit
    begin_vertex = matched_minimum_spanning_tree[0][0]
    E_P = [n[begin_vertex][0]]

    while len(matched_minimum_spanning_tree) !=0:
        for i, v in enumerate(E_P):
            if len(n[v]) !=0:
                break

        while len(n[v]) !=0:
            w = n[v][0]

            remove_duplicate_edge_from_matched_minimum_spanning_tree_fun(matched_minimum_spanning_tree, v, w)

            del n[v][(n[v].index(w))]
            del n[w][(n[w].index(v))]

            i += 0
            E_P.insert(i, w)

            v = w

    return E_P


def remove_duplicate_edge_from_matched_minimum_spanning_tree_fun(matched_minimum_spanning_tree, k1, k2):

    for i, item in enumerate(matched_minimum_spanning_tree):
        if (item[0] == k2 and item[1] == k1) or (item[0] == k1 and item[1] == k2):
            del matched_minimum_spanning_tree[i]

    return matched_minimum_spanning_tree



def find_minimum_weight_matching_slow(graph):
    # Graph should have an even number of vertices to have a perfect
    # matching.
    assert len(graph) % 2 == 0

    import itertools

    def score(matching):
        result = 0
        for u, v in matching:
            result += graph[u][v]
        return result

    best = None
    for permutation in itertools.permutations(range(len(graph))):
        matching = []
        for i in range(0, len(graph), 2):
            matching.append((permutation[i], permutation[i + 1]))
        if best is None or score(matching) < score(best):
            best = matching

    return best

def examples():
    # NB: There are many optimal solutions for TSP instances below, so
    # treat these asserts just as examples of a possible program behaviour.

    # We start at point 0 with coordinates (0, 0), then go to point 1
    # with coordinates (2, 2) and then return to point 0.
    assert solve_tsp([(0, 0), (2, 2)]) == [0, 1, 0]

    # Here we have four points in the corners of a unit square.
    # One possible tour is (1, 0) -> (0, 0) -> (0, 1) -> (1, 1) -> (1, 0).
    assert solve_tsp([(1, 1), (0, 0), (1, 0), (0, 1)]) == [2, 1, 3, 0, 2]

    # Examples of find_minimum_weight_matching_slow (and find_minimum_weight_matching) usage.

    # Here we have a graph with two vertices and one edge. The single possible perfect matching
    # is just a (0, 1) edge of a graph.
    assert find_minimum_weight_matching_slow([[0, 1], [1, 0]]) == [(0, 1)]

    # In a graph below there are two edges with weight 1 and all the other edges have weight 2.
    # The only possible way to obtain a perfect matching of weight 2 is to select both of the edges
    # with weight 1.
    assert find_minimum_weight_matching_slow([
        [0, 2, 1, 2],
        [2, 0, 2, 1],
        [1, 2, 0, 2],
        [2, 1, 2, 1],
    ]) == [(0, 2), (1, 3)]
    
    
    
#Define the points here
points = [
        [0, 2, 1, 2],
        [2, 0, 2, 1],
        [1, 2, 0, 2],
        [2, 1, 2, 1],
    ]


# MAIN ---------------------------------------------------------

def main(points):
    output = solve_tsp(points)
    return output

if __name__ == '__main__':
    main(points)


# In[8]:


#points 1
points = [(0, 0), (2, 2)]


if __name__ == '__main__':
    main(points)

