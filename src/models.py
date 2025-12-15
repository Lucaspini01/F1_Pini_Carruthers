from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

def build_preprocessor(legal_features_num, legal_features_cat):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), legal_features_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), legal_features_cat),
        ]
    )

def build_gb_fe2_baseline(legal_features_num, legal_features_cat):
    preprocessor = build_preprocessor(legal_features_num, legal_features_cat)
    gb = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", gb),
    ])

def build_gb_fe2_tuned(legal_features_num, legal_features_cat):
    preprocessor = build_preprocessor(legal_features_num, legal_features_cat)
    gb = GradientBoostingRegressor(
        random_state=42,
        
        # Parametros obtenidos por tuning con Optuna:
        n_estimators=679,       
        learning_rate=0.08698283488468485,    
        max_depth= 5,          
        subsample=0.6722054568397781,          
        min_samples_split=7,  
        min_samples_leaf=7,   
    )
    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", gb),
    ])
