from typing import Optional, List, Tuple, Dict

class SkNode:
    """
    Represents a node in a decision tree, mimicking scikit-learn's decision tree structure.
    
    Attributes:
        label: Unique identifier for the node.
        left: Left child node.
        right: Right child node.
        feature: Index of the feature used for splitting at this node.
        threshold: Threshold value for the split.
        parent: Parent node.
        is_leaf: True if this node is a leaf.
        is_left_child: True if this node is the left child of its parent.
        is_right_child: True if this node is the right child of its parent.
        prediction: For leaf nodes, the predicted class label.
        n_classes: Total number of classes (set from the tree).
        n_features: Total number of features (set from the tree).
    """
    def __init__(self, label: Optional[int] = None):
        self.label: Optional[int] = label
        self.left: Optional['SkNode'] = None
        self.right: Optional['SkNode'] = None
        self.feature: Optional[int] = None
        self.threshold: Optional[float] = None
        self.parent: Optional['SkNode'] = None
        self.is_leaf: bool = False
        self.is_left_child: bool = False
        self.is_right_child: bool = False
        self.prediction: Optional[int] = None
        self.n_classes: Optional[int] = None
        self.n_features: Optional[int] = None

    def get_constraints(self) -> Tuple[List[float], List[float]]:
        """
        Recursively traverses from this node up to the root to determine the constraints
        (minimum and maximum allowable values) for each feature based on the decisions made.

        Returns:
            A tuple (mins, maxs) where:
                - mins: A list of minimum bounds for each feature.
                - maxs: A list of maximum bounds for each feature.
        """
        # Initialize constraints with unbounded ranges.
        mins = [float('-inf')] * self.n_features
        maxs = [float('inf')] * self.n_features

        if self.parent is not None:
            # Get constraints from the parent node.
            mins, maxs = self.parent.get_constraints()
            split_feature = self.parent.feature
            # Adjust the constraint based on whether this node is a left or right child.
            if self.is_left_child:
                maxs[split_feature] = self.parent.threshold
            elif self.is_right_child:
                mins[split_feature] = self.parent.threshold
        return mins, maxs

def create_sk_nodes(sk_model) -> Dict[int, SkNode]:
    """
    Converts a fitted scikit-learn decision tree model into a dictionary of SkNodes.
    
    Args:
        sk_model: A fitted scikit-learn decision tree model.
    
    Returns:
        A dictionary mapping node indices to their corresponding SkNode objects.
    """
    # Access the underlying tree structure.
    tree = sk_model.tree_
    num_nodes = tree.node_count
    n_classes = tree.n_classes[0]
    n_features = tree.n_features

    # Create a dictionary of SkNode objects indexed by node id.
    nodes = {i: SkNode(label=i) for i in range(num_nodes)}

    for i in range(num_nodes):
        # Set common attributes.
        nodes[i].n_classes = n_classes
        nodes[i].n_features = n_features

        if tree.feature[i] >= 0:  # Non-leaf node
            left_index = tree.children_left[i]
            right_index = tree.children_right[i]

            # Link left child.
            nodes[i].left = nodes[left_index]
            nodes[left_index].parent = nodes[i]
            nodes[left_index].is_left_child = True

            # Link right child.
            nodes[i].right = nodes[right_index]
            nodes[right_index].parent = nodes[i]
            nodes[right_index].is_right_child = True

            # Set split parameters.
            nodes[i].feature = tree.feature[i]
            nodes[i].threshold = tree.threshold[i]
        else:
            # For leaf nodes, mark as leaf and determine the predicted class.
            nodes[i].is_leaf = True
            # Reshape the value array to (num_nodes, n_classes) and find the index of the maximum value.
            nodes[i].prediction = tree.value.reshape((-1, n_classes))[i].argmax()

    return nodes
