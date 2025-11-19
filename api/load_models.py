"""
LOAD MODELS PIPELINE
Charge tous les objets nécessaires pour la prédiction :
- Prophet
- SARIMAX
- LightGBM
- Stacking Model
- Scalers
- Features configurations
- Training metadata
"""

# pylint: disable=missing-function-docstring

import json
import pickle
from pathlib import Path
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

BASE_DIR = Path(__file__).resolve().parent  # root du projet


# ==============================
# 1. CHARGEMENT DES SCALERS
# ==============================

def load_scalers():
    scaler_path = BASE_DIR / "scaler" / "exog_scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print("✓ Scaler chargé")
    return scaler


# ===================================
# 2. CHARGEMENT DES FEATURES CONFIG
# ===================================

def load_feature_config():
    config_path = BASE_DIR / "metadata" / "features_config.json"
    with open(config_path, "rb") as f:
        features_config = json.load(f)
    print("✓ Features config chargé")
    return features_config


# ========================================
# 3. CHARGEMENT DES MÉTADONNÉES TRAINING
# ========================================

def load_training_info():
    metadata_path = BASE_DIR / "metadata" / "training_info.json"
    with open(metadata_path, "rb") as f:
        training_info = json.load(f)
    print("✓ Training info chargé")
    return training_info


# ==============================
# 4. CHARGEMENT DES MODÈLES
# ==============================

def load_prophet():
    model_path = BASE_DIR / "models" / "prophet_model.pkl"
    with open(model_path, "rb") as f:
        prophet_model = pickle.load(f)
    print("✓ Prophet chargé")
    return prophet_model


def load_sarimax():
    model_path = BASE_DIR / "models" / "sarimax_model_v1.pkl"
    sarimax_model = SARIMAXResults.load(model_path)
    print("✓ SARIMAX chargé")
    return sarimax_model


def load_lightgbm():
    model_path = BASE_DIR / "models" / "lightgbm_model_v1.txt"
    lgb_model = lgb.Booster(model_file=str(model_path))
    print("✓ LightGBM chargé")
    return lgb_model


def load_stacking():
    stacking_path = BASE_DIR / "models" / "stacking_model_v1.pkl"
    with open(stacking_path, "rb") as f:
        stacking_model = pickle.load(f)
    print("✓ Stacking model chargé")
    return stacking_model


# ==============================
# 5. PIPELINE GLOBAL DE CHARGEMENT
# ==============================
def load_all_models():
    print("\n=== CHARGEMENT COMPLET DES MODÈLES ===\n")

    scaler = load_scalers()
    features_config = load_feature_config()
    training_info = load_training_info()

    prophet_model = load_prophet()
    sarimax_model = load_sarimax()
    lgb_model = load_lightgbm()
    stacking_model = load_stacking()

    print("\n✓✓✓ Tous les modèles et configurations ont été chargés ✓✓✓\n")

    return {
        "scaler": scaler,
        "features_config": features_config,
        "training_info": training_info,
        "prophet": prophet_model,
        "sarimax": sarimax_model,
        "lgbm": lgb_model,
        "stacking": stacking_model
    }


# ==============================
# 6. TEST ÉXÉCUTABLE
# ==============================
if __name__ == "__main__":
    objects = load_all_models()
    print(objects)
