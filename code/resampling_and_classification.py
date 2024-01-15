from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, AllKNN, CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, TomekLinks, InstanceHardnessThreshold, NearMiss, OneSidedSelection
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import NeighbourhoodCleaningRule


resampling_techniques = {
        "RO": RandomOverSampler(random_state=0),
        "SM": SMOTE(random_state=0),
        "BS": BorderlineSMOTE(random_state=0),
        "AD": ADASYN(random_state=0),
        "ST": SMOTETomek(),
        "CN": CondensedNearestNeighbour(random_state=0),
        "RU": RandomUnderSampler(random_state=0),
        "AL": AllKNN(),
        "EN": EditedNearestNeighbours(),
        "RE": RepeatedEditedNearestNeighbours(),
        "TM": TomekLinks(),
        "IH": InstanceHardnessThreshold(),
        "OS": OneSidedSelection(),
        "NM": NearMiss(version=1),
        "SE": SMOTEENN(random_state=0),
        "NC": NeighbourhoodCleaningRule(),
        "Original": None
    }

# 5 Oversampling, 9 Undersampling, 2 Hybrid sampling
# "Random Over-Sampling (RO)": RandomOverSampler(random_state=0),
# "SMOTE Over-Sampling (SM)": SMOTE(random_state=0),
# "BorderlineSMOTE Over-Sampling (BS)": BorderlineSMOTE(random_state=0),
# "ADASYN Over-Sampling (AD)": ADASYN(random_state=0),
# "Synthetic minority oversampling-Tomek’s link Over-Sampling (ST)": SMOTETomek(),

# "Condensed nearest neighbours Under-Sampling (CN)": CondensedNearestNeighbour(random_state=0),
# "Random Under-Sampling (RU)": RandomUnderSampler(random_state=0),
# "All k-nearest neighbours Under-Sampling (AL)": AllKNN(),
# "Edited nearest neighbours Under-Sampling (EN)": EditedNearestNeighbours(),
# "Repeated edited nearest neighbours Under-Sampling (RE)": RepeatedEditedNearestNeighbours(),
# "Tomek links Under-Sampling(TM)": TomekLinks(),
# "Instance hardness threshold Under-Sampling (IH)": InstanceHardnessThreshold(),
# "One-sided selection Under-Sampling (OS)": OneSidedSelection(),
# "NearMiss Under-Sampling(NM)": NearMiss(version=1),

# "Synthetic minority oversampling-Edited nearest neighbour Hybrid-Sampling (SE)": SMOTEENN(random_state=0),
# "Neighbourhood cleaning rule Hybrid-Sampling(NC)": NeighbourhoodCleaningRule(),