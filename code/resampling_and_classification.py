from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, AllKNN, CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, TomekLinks, InstanceHardnessThreshold, NearMiss, OneSidedSelection
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import NeighbourhoodCleaningRule


resampling_techniques = {
        "Random Over-Sampling (RO)": RandomOverSampler(random_state=0),
        "SMOTE Over-Sampling (SM)": SMOTE(random_state=0),
        "BorderlineSMOTE Over-Sampling (BS)": BorderlineSMOTE(random_state=0),
        "Random Under-Sampling (RU)": RandomUnderSampler(random_state=0),
        "All k-nearest neighbours Under-Sampling (AL)": AllKNN(),
        "Condensed nearest neighbours Under-Sampling (CN)": CondensedNearestNeighbour(random_state=0),
        "Edited nearest neighbours Under-Sampling (EN)": EditedNearestNeighbours(),
        "Repeated edited nearest neighbours Under-Sampling": RepeatedEditedNearestNeighbours(),
        "Synthetic minority oversampling-Edited nearest neighbour Hybrid-Sampling (SE)": SMOTEENN(random_state=0),
        "Tomek links Under-Sampling(TM)": TomekLinks(),
        "Synthetic minority oversampling-Tomek’s link Over-Sampling (ST)": SMOTETomek(),
        "Instance hardness threshold Under-Sampling (IH)": InstanceHardnessThreshold(),
        "Neighbourhood cleaning rule Hybrid-Sampling(NC)": NeighbourhoodCleaningRule(),
        "One-sided selection Under-Sampling (OS)": OneSidedSelection(),
        "ADASYN Over-Sampling (AD)": ADASYN(random_state=0),
        "NearMiss Under-Sampling(NM)": NearMiss(version=1),
        "No Filter": None
    }

# 5 Oversampling, 9 Undersampling, 2 Hybrid sampling