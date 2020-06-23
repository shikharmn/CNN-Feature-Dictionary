# CNN-Feature-Dictionary

An implementation of the CNN Feature Dictionary algorithm, for the MV Tech dataset for anomaly detection in texture based datasets.

Results for the MV Tech dataset: https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/mvtec_ad.pdf

The CNN Feature Dictionary algorithm: https://www.researchgate.net/publication/322476121_Anomaly_Detection_in_Nanofibrous_Materials_by_CNN-Based_Self-Similarity

Instead of ResNet-18, this implementation uses ResNext-101.

Other hyperparameters include patch size being 16x16, a stride of 4 and 10 clusters in building the feature dictionary using K-Means.

Change the path in data_helpers.py and the hook function in model_helpers.py to modify the path and model respectively.
