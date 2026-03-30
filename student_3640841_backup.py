############ CODE BLOCK 0 ################
#^ DO NOT CHANGE THIS LINE

import numpy as np
import pandas as pd
import pygraphviz as pgv
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.metrics import accuracy_score

############ CODE BLOCK 1 ################
#^ DO NOT CHANGE THIS LINE

def gini(labels):
    if len(labels) == 0:
        return 0
    frequencies = labels.value_counts().values 
    probabilities = np.asarray([f / len(labels) for f in frequencies])  # everything including the first class
    gini_impurity = 1-sum(probabilities**2)
    return gini_impurity

############ CODE BLOCK 2 ################
#^ DO NOT CHANGE THIS LINE

def entropy(labels):
    if len(labels) == 0:
        return 0
    frequencies = labels.value_counts().values 
    probabilities = np.asarray([f / len(labels) for f in frequencies])  # everything including the first class
    entropy = -sum(probabilities*np.log2(probabilities))
    return entropy

############ CODE BLOCK 3 ################
#^ DO NOT CHANGE THIS LINE

class DTree:
    def __init__(self, metric):
        """Set up a new tree.

        We use the `metric` parameter to supply an impurity measure such as Gini or Entropy.
        The other class variables should be set by the "fit" method.
        """
        self._metric = metric  # what are we measuring impurity with? (Gini, Entropy, Minority Class...)
        self._samples = None  # how many training samples reached this node?
        self._distribution = []  # what was the class distribution in this node?
        self._label = None  # What was the majority class of training samples that reached this node?
        self._leaf_id = None
        self._impurity = None  # what was the impurity at this node?
        self._split = False  # if False, then this is a leaf. If you branch from this node, use this to store the name of the feature you're splitting on.
        self._yes = None  # Holds the "yes" DTree object; None if this is still a leaf node
        self._no = None # Holds the "no" DTree object; None if this is still a leaf node



    def _best_split(self, features, labels):
        """ Determine the best feature to split on.

        :param features: a pd.DataFrame with named training feature columns
        :param labels: a pd.Series or pd.DataFrame with training labels
        :return: `best_so_far` is a string with the name of the best feature,
        and `best_so_far_impurity` is the impurity on that feature

        For each candidate feature the weighted impurity of the "yes" and "no"
        instances for that feature are computed using self._metric.

        We select the feature with the lowest weighted impurity.
        """
        #alldata= pd.concat([features, labels], axis=1)

        #splits = [(alldata[features[feature]>=.5],alldata[features[feature]<.5]) for feature in features]

        impurities = []
        n_total = len(labels)



        for col in features.columns:
            # 1. Create the mask for the split
            mask = features[col] >= 0.5

            # Split the labels
            labels_yes = labels[mask]
            labels_no = labels[~mask]

            # Calculate weights 
            weight_yes = len(labels_yes) / n_total
            weight_no = len(labels_no) / n_total

            imp_yes = self._metric(labels_yes) 
            imp_no = self._metric(labels_no) 

            weighted_impurity = (weight_yes * imp_yes) + (weight_no * imp_no)
            impurities.append(weighted_impurity)

        #impurities = np.asarray([(len(split[0])*self._metric(split[0]))+len(split[1])*self._metric(split[1]) for split in splits])/len(alldata)


        #best = features.columns[np.argmin(impurities)]

        best_so_far = features.columns[np.argmin(impurities)]
        best_so_far_impurity = np.min(impurities)

        return best_so_far, best_so_far_impurity

    def fit(self, features, labels, all_classes=None):
        """ Generate a decision tree by recursively fitting & splitting them

        :param features: a pd.DataFrame with named training feature columns
        :param labels: a pd.Series or pd.DataFrame with training labels
        :return: Nothing.

        First this node is fitted as if it was a leaf node: the training majority label, number of samples,
        class distribution and impurity.

        Then we evaluate which feature might give the best split.

        If there is a best split that gives a lower weighed impurity of the child nodes than the impurity in this node,
        initialize the self._yes and self._no variables as new DTrees with the same metric.
        Then, split the training instance features & labels according to the best splitting feature found,
        and fit the Yes subtree with the instances that split to the True side,
        and the No subtree with the instances that are False according to the splitting feature.
        """
        if all_classes is None:
            self.all_classes = sorted(labels.unique())
        else:
            self.all_classes = all_classes

        self._samples = len(labels)  # how many training samples reached this node?


        counts_series = labels.value_counts().reindex(self.all_classes, fill_value=0)
        self._distribution = counts_series.values  # what was the class distribution in this node?
        self._label = counts_series.idxmax()# What was the majority class of training samples that reached this node?
        self._impurity = self._metric(labels)  # what was the impurity at this node?
        self._split = False  # if False, then this is a leaf. If you branch from this node, use this to store the name of the feature you're splitting on.
        self._yes = None  # Holds the "yes" DTree object; None if this is still a leaf node
        self._no = None # Holds the "no" DTree object; None if this is still a leaf node


        split, split_impurity = self._best_split(features, labels)  # Find the best split, if any

        shouldsplit = split_impurity < self._impurity
        if shouldsplit:
            self._split = split


            metric=self._metric
            yes_split_features = features[features[split]>.5]
            yes_split_labels = labels[features[split]>.5]
            self._yes = DTree(metric=metric)
            self._yes.fit(yes_split_features,yes_split_labels,all_classes=self.all_classes)


            no_split_features = features[features[split]<=.5]
            no_split_labels = labels[features[split]<=.5]
            self._no = DTree(metric=metric)
            self._no.fit(no_split_features,no_split_labels,all_classes=self.all_classes)




    def predict(self, features):
        """ Predict the labels of the instances based on the features

        :param features: pd.DataFrame of test features
        :return: predicted labels

        We start by initializing an array of labels where we naively predict this node's label.
        The datatype of this array is set to `object` because otherwise numpy
        might select the minimum needed string length for the current label, regardless of child labels.

        Then if this is not a leaf node, we overwrite those values with the values of Yes and No child nodes,
        based on the feature split in this node.
        """
        results = np.full(features.shape[0], self._label, dtype=object)  # object!!!
        if self._split:  # branch node; recursively replace predictions with child predictions
            yes_index = features[self._split] > 0.5
            results[yes_index] = self._yes.predict(features.loc[yes_index])
            results[~yes_index] = self._no.predict(features.loc[~yes_index])
        return results

    def predict_labeled(self, features, root = True):
        if root: 
            self.label_leaves()
        """ Predict the labels of the instances based on the features, 
        and add the label of in which leaf the instance lands.

        :param features: pd.DataFrame of test features
        :return: predicted labels

        We start by initializing an array of labels where we naively predict this node's label.
        The datatype of this array is set to `object` because otherwise numpy
        might select the minimum needed string length for the current label, regardless of child labels.

        Then if this is not a leaf node, we overwrite those values with the values of Yes and No child nodes,
        based on the feature split in this node.
        """

        results_labeled = np.empty((features.shape[0], 2), dtype=object)


        if self._split:  # branch node; recursively replace predictions with child predictions
            yes_index = features[self._split] > 0.5
            results_labeled[yes_index] = self._yes.predict_labeled(features.loc[yes_index],root=False)
            results_labeled[~yes_index] = self._no.predict_labeled(features.loc[~yes_index],root=False)
        else:
            results_labeled[:, 0] = self._label
            results_labeled[:, 1] = getattr(self, '_leaf_id', -1)

        return results_labeled

    def to_text(self, depth=0):
        if self._split:
            text = f'{"|   " * depth}|---{self._split} = no\n'
            text += self._no.to_text(depth=depth+1)
            text += f'{"|   " * depth}|---{self._split} = yes\n'
            text += self._yes.to_text(depth=depth+1)

        else:
            text = f'{"|   " * depth}|---{self._label} ({self._samples})\n'.upper()
        return text

    def label_leaves(self, current_count=None):
        if current_count is None:
            current_count = [1]
        """ Recursively assign a unique ID to each leaf node. """
        if self._leaf_id is not None: 
            return current_count[0]
        if not self._split:
            self._leaf_id = current_count[0]
            current_count[0] += 1
        else:
            # If it's a branch, keep going down
            self._no.label_leaves(current_count)
            self._yes.label_leaves(current_count)

        return current_count[0] 

    def to_graphviz(self, choice='', parent='R', graph=None, size='15,15'):
        details = f'\n\nimpurity = {self._impurity:.2f}\nsamples = {self._samples}\n{self._distribution}'
        if self._split:
            label = f'({self._label.lower()})'
        else:
            label = self._label.upper()
        if graph is None:  # root node
            graph = pgv.AGraph(directed=True)  # initialize the graph
            graph.graph_attr.update(size=size)
            graph.graph_attr.update(ratio='1.0')
        if self._split:  # branching nodes
            node_label = f'{label}\n{self._split.upper()}???{details}'  # display name
            graph.add_node(n=parent+choice, label=node_label, shape='diamond')
            self._yes.to_graphviz(choice='yes', parent=parent+choice, graph=graph)
            self._no.to_graphviz(choice='no', parent=parent+choice, graph=graph)
        else:  # leaf node
            node_label = f'{label}{details}'  # display name
            graph.node_attr.update(name=parent+choice, label=node_label, shape='rectangle')
        if choice != '':
            graph.add_edge(parent, parent + choice, label=choice)  # draw arrow from parent to this one
        return graph

    def to_graphviz_labeled(self, choice='', parent='R', graph=None, size='15,15', root=True):
        if root: self.label_leaves()


        details = f'\n\nimpurity = {self._impurity:.2f}\nsamples = {self._samples}\n{self._distribution}'


        # Check if leaf_id exists, otherwise use an empty string
        leaf_id_str = f"LEAF #{self._leaf_id}\n" if hasattr(self, '_leaf_id') else ""

        if self._split:
            label = f'({self._label.lower()})'
        else:
            label = self._label.upper()

        if graph is None:  
            graph = pgv.AGraph(directed=True)
            graph.graph_attr.update(size=size)
            graph.graph_attr.update(ratio='1.0')

        if self._split:  
            node_label = f'{label}\n{self._split.upper()}???{details}'
            graph.add_node(n=parent+choice, label=node_label, shape='diamond')


            self._yes.to_graphviz_labeled(choice='yes', parent=parent+choice, graph=graph,root=False)
            self._no.to_graphviz_labeled(choice='no', parent=parent+choice, graph=graph,root=False)
        else:  
            node_label = f'{leaf_id_str}{label}{details}'
            graph.add_node(n=parent+choice, label=node_label, shape='rectangle')

        if choice != '':
            graph.add_edge(parent, parent + choice, label=choice)

        return graph

