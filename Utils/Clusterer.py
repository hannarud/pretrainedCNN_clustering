import os
import pickle
import csv

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans, MeanShift, AffinityPropagation, Birch, DBSCAN
from sklearn.metrics import normalized_mutual_info_score as nmi
from Metrics.purity import purity


class Clusterer:
    def __init__(self, dataset, cnn_architecture, layer, clustering_algorithm, n_classes=0):
        self.dataset_path = os.path.join('.', 'data', dataset)
        self.dataset_feat_path = os.path.join(self.dataset_path, 'features', str('%s_%s' % (cnn_architecture, layer)))

        self.dataset_feature_names = os.listdir(self.dataset_feat_path)
        self.n_files = len(self.dataset_feature_names)
        
        self.get_n_classes(n_classes)
        print("Number of classes: %d" % self.n_classes)
        
        self.get_algorithm(clustering_algorithm)
        print("Algorithm: " + str(self.algorithm).split("(")[0])
        
        self.get_features()
        print("Features shape: " + str(self.features.shape))
            
    def get_n_classes(self, n_classes):
        if os.path.exists(os.path.join(self.dataset_path, "true_labels.csv")):
            with open(os.path.join(self.dataset_path, "true_labels.csv"), mode='r') as true_lab_file:
                reader = csv.reader(true_lab_file, delimiter=';')
                next(reader, None) # Skip header
                self.true_labels = {rows[0]: rows[1] for rows in reader}

            self.n_classes = len(list(set(self.true_labels.values())))
            integer_classes = list(range(self.n_classes))
            self.original_classes_to_int = dict(zip(list(set(self.true_labels.values())), integer_classes))

            self.true_labels = {tr_lab:self.original_classes_to_int[self.true_labels[tr_lab]] for tr_lab in self.true_labels.keys()}
        elif n_classes != 0:
            self.n_classes = n_classes
            self.true_labels = 0
        else:
            print("Error: %s folder must contain a true_labels.csv file OR n_classes must be a positive integer" % self.dataset_path)
            return
    
    def get_algorithm(self, clustering_algorithm):
        if clustering_algorithm == "kmeans":
            self.algorithm = KMeans(self.n_classes)
        elif clustering_algorithm == "mb_kmeans":
            self.algorithm = MiniBatchKMeans(self.n_classes)
        elif clustering_algorithm == "affinity_prop":
            self.algorithm = AffinityPropagation()
        elif clustering_algorithm == "mean_shift":
            self.algorithm = MeanShift()
        elif clustering_algorithm == "agglomerative":
            self.algorithm = AgglomerativeClustering(self.n_classes)
        elif clustering_algorithm == "birch":
            self.algorithm = Birch(self.n_classes)
        elif clustering_algorithm == "dbscan":
            self.algorithm = DBSCAN()
        else:
            print("Error: This clustering algorithm is not available. Choose among the following options: 'kmeans', 'mb_kmeans', 'affinity_prop', 'mean_shift', 'agglomerative', 'birch', 'dbscan'")
        
    def get_features(self):
        self.features = []
        for i in range(self.n_files):
            file = open(os.path.join(self.dataset_feat_path, self.dataset_feature_names[i]), "rb")
            self.features.append(pickle.load(file))
            file.close()
        self.features = np.array(self.features)
    
    def cluster(self):
        print("Clustering ...")
        self.predicted_labels = self.algorithm.fit_predict(self.features)
    
    def evaluate(self, metric):
        if self.true_labels == 0:
            print("Error: A true_labels.csv file is needed")
            return

        true_labels = list(self.true_labels.values())
        
        if metric == "nmi":
            print("NMI: %f" % nmi(true_labels, self.predicted_labels))
        elif metric == "purity":
            print("Purity: %f" % purity(true_labels, self.predicted_labels))
        elif metric == "both":
            print("NMI: %f" % nmi(true_labels, self.predicted_labels))
            print("Purity: %f" % purity(true_labels, self.predicted_labels))
        else:
            print("Error: This metric is not available. Choose among the following options: 'nmi', 'purity', 'both'")
