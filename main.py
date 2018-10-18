from Utils.Feature_extractor import Feature_extractor
from Utils.Clusterer import Clusterer

dataset = "simpsons_and_peppa"
cnn_architecture = "vgg19"
layer = "fc2"
clustering_algorithm = "agglomerative"
metric = "both"

fe = Feature_extractor(dataset, cnn_architecture, layer)
fe.extract_and_save_features()
cl = Clusterer(dataset, cnn_architecture, layer, clustering_algorithm)
cl.cluster()
print(cl.original_classes_to_int)
predicted_labels = cl.predicted_labels
print("Shape predicted labels: %s" % str(predicted_labels.shape))
cl.evaluate(metric)
