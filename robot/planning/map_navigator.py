#!/usr/bin/env python3
"""
Map Navigator & Shortest Path Finder
====================================
Tìm đường đi ngắn nhất trên đồ thị giao lộ cho bài thi Smart City (Dijkstra, A*).
"""

import json
import networkx as nx


class ShortestPathFinder:
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.start_node = self._find_node_by_type('start')
        self.end_node = self._find_node_by_type('end')
        self.graph = self._build_graph()  # {u: {v: (weight, label)}}
        self.dinh = self._convert_edges_to_nodes()

    def _find_node_by_type(self, node_type):
        for node in self.data['nodes']:
            if node['type'] == node_type:
                return node['id']
        return None

    def _build_graph(self):
        graph = {}
        for edge in self.data['edges']:
            u = edge['source']
            v = edge['target']
            w = 1
            if u not in graph:
                graph[u] = {}
            graph[u][v] = (w, edge.get('label', 'straight'))
        return graph

    def _convert_edges_to_nodes(self):
        cost_xoay = 1
        dinh = {}
        graph = self.graph
        for edge in self.data['edges']:
            u = edge['source']
            v = edge['target']
            huong_u_v = graph[u][v][1]
            for v2, value in graph.get(v, {}).items():
                if v2 == u:
                    continue
                huong_v_v2 = value[1]
                cost = value[0]
                if huong_u_v != huong_v_v2:
                    cost += cost_xoay
                if (u, v) not in dinh:
                    dinh[(u, v)] = {}
                dinh[(u, v)][(v, v2)] = cost
        return dinh

    def convert_to_graph(self, dinh):
        G = nx.DiGraph()
        for u in dinh:
            for v in dinh[u]:
                cost = dinh[u][v]
                G.add_edge(u, v, weight=cost)
        return G

    def find(self):
        start_node = self.start_node
        end_node = self.end_node
        return self.find_between(start_node, end_node)

    def find_between(self, start_node, end_node):
        result_path = []
        res_cost = float('inf')
        for u in self.graph.get(start_node, {}):
            for v in self.graph:
                if end_node in self.graph[v]:
                    path, cost = self.shortest_path((start_node, u), (v, end_node))
                    if not path:
                        continue
                    cost += self.graph[start_node][u][0]
                    if cost < res_cost:
                        result_path = path
                        res_cost = cost
        result = []
        for u, v in result_path:
            result.append((u, v, self.graph[u][v][1]))

        return result, res_cost

    def shortest_path(self, start, end):
        G = self.convert_to_graph(self.dinh)
        try:
            path = nx.dijkstra_path(G, source=start, target=end, weight='weight')
            length = nx.dijkstra_path_length(G, source=start, target=end, weight='weight')
            return path, length
        except nx.NetworkXNoPath:
            return None, float('inf')


class MapNavigator:
    """
    Bộ điều hướng bản đồ giao lộ Smart City.
    """
    def __init__(self, json_path):
        self.finder = ShortestPathFinder(json_path)
        self.node_lookup = {n['id']: n for n in self.finder.data.get('nodes', [])}

    @property
    def start_node(self):
        return self.finder.start_node

    @property
    def end_node(self):
        return self.finder.end_node

    def _build_nx_graph(self):
        G = nx.DiGraph()
        for u, nbrs in self.finder.graph.items():
            for v, (w, label) in nbrs.items():
                G.add_edge(u, v, weight=w, label=label)
        return G

    def _heuristic(self, a, b):
        na = self.node_lookup.get(a)
        nb = self.node_lookup.get(b)
        if not na or not nb:
            return 0
        try:
            dx = na.get('x', 0) - nb.get('x', 0)
            dy = na.get('y', 0) - nb.get('y', 0)
            return (dx * dx + dy * dy) ** 0.5
        except Exception:
            return 0

    def find_path(self, start, end, banned_edges=None):
        """Trả về danh sách các node id từ start đến end (có hỗ trợ cấm cạnh)."""
        G = self._build_nx_graph()
        if banned_edges:
            for edge in banned_edges:
                try:
                    if G.has_edge(edge[0], edge[1]):
                        G.remove_edge(edge[0], edge[1])
                except Exception:
                    pass
        try:
            path = nx.astar_path(G, source=start, target=end, heuristic=self._heuristic)
            return path
        except Exception:
            return None

    def get_next_direction_label(self, current_node_id, path):
        """Trả về nhãn hướng rẽ (straight/turn_left/turn_right) đến node tiếp theo."""
        if not path or current_node_id not in path:
            return None
        idx = path.index(current_node_id)
        if idx + 1 >= len(path):
            return None
        next_node = path[idx + 1]
        nbrs = self.finder.graph.get(current_node_id, {})
        entry = nbrs.get(next_node)
        if entry:
            return entry[1]
        return None

    def get_neighbor_by_direction(self, current_node_id, direction_label):
        for v, (w, label) in self.finder.graph.get(current_node_id, {}).items():
            if label == direction_label:
                return v
        return None
