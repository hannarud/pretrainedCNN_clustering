from Utils.Feature_extractor import Feature_extractor
from Utils.Clusterer import Clusterer

dataset = "fencing"
cnn_architecture = "vgg19"  # Options available: 'vgg16', 'vgg19', 'resnet', 'xception', 'inception'
layer = "fc2"
clustering_algorithm = "agglomerative"  # Options available: 'kmeans', 'mb_kmeans', 'affinity_prop', 'mean_shift', 'agglomerative', 'birch', 'dbscan'
metric = "all"  # Options available: "nmi", "purity", "confusion matrix", "all" (= "nmi'+"purity"+"confusion matrix")

fe = Feature_extractor(dataset, cnn_architecture, layer)
fe.extract_and_save_features()
cl = Clusterer(dataset, cnn_architecture, layer, clustering_algorithm)
cl.cluster()
print(cl.original_classes_to_int)
predicted_labels = cl.predicted_labels
print("Shape predicted labels: %s" % str(predicted_labels.shape))
cl.evaluate(metric)
cl.plot_features()
