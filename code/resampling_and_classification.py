from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from LSTMencoder_pytorch import LSTM, SequenceDataset


resampling_techniques = {
        "Random Over-Sampling": RandomOverSampler(random_state=0),
        "Random Under-Sampling": RandomUnderSampler(random_state=0),
        "SMOTE": SMOTE(random_state=0),
        "ADASYN": ADASYN(random_state=0),
        "NearMiss": NearMiss(version=1),
        "No Filter": None
    }
