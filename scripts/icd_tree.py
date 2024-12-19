import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

class ICDTree:
    def __init__(self, dataframe):
        """
        Initialize the ICDTree with a pandas DataFrame.

        Parameters:
            dataframe (pd.DataFrame): DataFrame containing ICD codes with the columns:
                - coding: The coding system (e.g., ICD10)
                - meaning: Description of the node
                - node_id: Unique identifier for the node
                - parent_id: Identifier of the parent node (or null for the root node)
                - selectable: Whether the node is selectable (Y/N)
        """
        self.graph = nx.DiGraph()
        
        # Create nodes and edges from the DataFrame
        self.coding_to_node_id = {}
        for _, row in dataframe.iterrows():
            node_id = row['node_id']
            self.graph.add_node(node_id, coding=row['coding'], meaning=row['meaning'], selectable=row['selectable'])
            self.coding_to_node_id[row['coding']] = node_id
            
            if pd.notna(row['parent_id']):
                self.graph.add_edge(row['parent_id'], node_id, weight=1.0)

    def initialize_edge_weights(self, weights):
        """
        Initialize edge weights based on the depth of child nodes.

        Parameters:
            weights (list of float): A list of weights where the index corresponds to the depth.
                                    For example, weights[0] is for depth 0 → 1, weights[1] for depth 1 → 2, etc.
        """
        # Get the root node
        root = self.get_root_nodes()[0]

        # Recursive function to assign weights
        def assign_weights_by_depth(node_id, current_depth):
            for child in self.get_children(node_id):
                # Determine the weight based on the depth
                weight = weights[min(current_depth, len(weights) - 1)]
                
                # Update the edge weight
                self.graph.edges[node_id, child]['weight'] = weight
                
                # Recursively assign weights to the children
                assign_weights_by_depth(child, current_depth + 1)

        # Start assigning weights from the root at depth 0
        assign_weights_by_depth(root, 0)

    def initialize_edge_weights_linear(self, initial_weight, final_weight):
        """
        Initialize edge weights linearly from an initial weight to a final weight based on the tree's depth.

        Parameters:
            initial_weight (float): The weight for the first depth (root to child).
            final_weight (float): The weight for the last depth (deepest leaf nodes).
        """
        # Get the maximum depth of the tree
        max_depth = self.get_max_depth()

        # Calculate weights for each depth
        weights = [
            initial_weight + (final_weight - initial_weight) * (depth / max_depth)
            for depth in range(max_depth + 1)
        ]

        # Recursive function to assign weights
        def assign_weights_linearly(node_id, current_depth):
            for child in self.get_children(node_id):
                # Use the weight corresponding to the current depth
                weight = weights[current_depth]
                
                # Update the edge weight
                self.graph.edges[node_id, child]['weight'] = weight
                
                # Recursively assign weights to the children
                assign_weights_linearly(child, current_depth + 1)

        # Get the root node
        root = self.get_root_nodes()[0]

        # Start assigning weights from the root at depth 0
        assign_weights_linearly(root, 0)

    def get_root_nodes(self):
        """Return the root nodes of the tree."""
        return [node for node, degree in self.graph.in_degree() if degree == 0]

    def get_children(self, node_id):
        """Return the children of a given node."""
        return list(self.graph.successors(node_id))

    def get_parent(self, node_id):
        """Return the parent of a given node. If no parent exists, return None."""
        parents = list(self.graph.predecessors(node_id))
        return parents[0] if parents else None

    def is_leaf(self, node_id):
        """Return True if the given node is a leaf, False otherwise."""
        return self.graph.out_degree(node_id) == 0

    def get_node_info(self, node_id):
        """Return the information (attributes) of a given node."""
        return self.graph.nodes[node_id]

    def get_node_id_from_coding(self, icd_code):
        """
        Get the node_id corresponding to a given ICD code.

        Parameters:
            icd_code (str): The coding of the node.

        Returns:
            int: The node_id corresponding to the ICD code.
        """
        if icd_code not in self.coding_to_node_id:
            raise ValueError(f"ICD code '{icd_code}' not found in the tree.")
        return self.coding_to_node_id[icd_code]
                    
    def get_max_depth(self):
        """
        Compute the maximum depth of the ICD tree, where the root has a depth of 0.

        Returns:
            int: The maximum depth of the tree.
        """
        # Get the single root node
        root = self.get_root_nodes()[0]

        # Function to recursively find the depth of a node
        def find_depth(node_id, current_depth):
            if self.is_leaf(node_id):
                return current_depth
            return max(find_depth(child, current_depth + 1) for child in self.get_children(node_id))

        # Compute the depth starting from the root with depth 0
        return find_depth(root, 0)
    
    def get_tree_distance(self, icd_code1, icd_code2):
        node_id1 = self.get_node_id_from_coding(icd_code1)
        node_id2 = self.get_node_id_from_coding(icd_code2)

        # Find the lowest common ancestor (LCA)
        lca = self.get_lca(node_id1, node_id2)
        if lca is None:
            return None

        # Distance from node1 to LCA
        dist1 = self.distance_to_ancestor(node_id1, lca)

        # Distance from node2 to LCA
        dist2 = self.distance_to_ancestor(node_id2, lca)

        return dist1 + dist2
    
    def visualize_path(self, icd_code1, icd_code2):
        node_id1 = self.get_node_id_from_coding(icd_code1)
        node_id2 = self.get_node_id_from_coding(icd_code2)

        lca = self.get_lca(node_id1, node_id2)
        if lca is None:
            print(f"No path exists between ICD codes '{icd_code1}' and '{icd_code2}'.")
            return

        # Get paths from each node up to the LCA
        path1 = self.path_to_ancestor(node_id1, lca)
        path2 = self.path_to_ancestor(node_id2, lca)

        # Combine unique nodes from both paths
        nodes = set(path1 + path2)
        edges = []

        # For each node (except the LCA), record its parent edge
        for n in nodes:
            if n != lca:
                p = self.get_parent(n)
                if p in nodes:
                    edges.append((p, n))

        # Construct a new graph with just these nodes and edges
        G_sub = nx.DiGraph()
        G_sub.add_nodes_from(nodes)
        G_sub.add_edges_from(edges)

        # Use graphviz_layout for a hierarchical layout
        pos = nx.nx_agraph.graphviz_layout(G_sub, prog="dot")

        plt.figure(figsize=(12, 8))
        nx.draw(G_sub, pos, node_color='red', edge_color='blue', with_labels=False, node_size=500)
        nx.draw_networkx_labels(G_sub, pos,
            labels={node: self.graph.nodes[node]['meaning'] for node in nodes},
            font_color='black')

        edge_labels = {}
        for (u, v) in edges:
            if 'weight' in self.graph.edges[u, v]:
                edge_labels[(u, v)] = f"{self.graph.edges[u, v]['weight']:.2f}"
        nx.draw_networkx_edge_labels(G_sub, pos, edge_labels=edge_labels)

        plt.title(f"Path between {icd_code1} and {icd_code2}")
        plt.show()


    def path_to_ancestor(self, node_id, ancestor_id):
        path = []
        current = node_id
        while current is not None:
            path.append(current)
            if current == ancestor_id:
                break
            current = self.get_parent(current)
        return path

    def distance_to_ancestor(self, node_id, ancestor_id):
        dist = 0.0
        current = node_id
        while current != ancestor_id:
            parent = self.get_parent(current)
            dist += self.graph.edges[parent, current]['weight']
            current = parent
        return dist

    def get_lca(self, node_id1, node_id2):
        ancestors1 = set(self.path_to_ancestor(node_id1, self.get_root_nodes()[0]))
        current = node_id2
        while current is not None:
            if current in ancestors1:
                return current
            current = self.get_parent(current)
        return None