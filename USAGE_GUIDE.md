
# GUIDE DE RÉUTILISATION DES MODÈLES

Ce guide explique comment :

- Charger les modèles sauvegardés  
- Générer les features futures  
- Produire les prévisions  
- Utiliser le stacking final 

## 1. Charger les modèles
```python
import pickle
import json
import pandas as pd
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

# Charger configs
with open("metadata/features_config.json", "r") as f:
  features_config = json.load(f)
with open("metadata/model_config.json", "r") as f:
  model_config = json.load(f)
with open("metadata/training_info.json", "r") as f:
  training_info = json.load(f)

# Scaler exogènes
with open("scaler/exog_scaler.pkl", "rb") as f:
  exog_scaler = pickle.load(f)

# Prophet
with open("models/prophet_model.pkl", "rb") as f:
  prophet_model = pickle.load(f)

# SARIMAX
sarimax_model = SARIMAXResults.load("models/sarimax_model_v1.pickle")

# LightGBM
lgb_model = lgb.Booster(model_file="models/lightgbm_model_v1.txt")

# Stacking
with open("models/stacking_model_v1.pkl", "rb") as f:
  stacking_model = pickle.load(f)
```

## 2. Préparation des futures dates
```python
future_dates = pd.date_range(
  start=training_info['date_max'] + pd.Timedelta(days=1),
  periods=model_config["horizon"],
  freq="D"
)

future_df = pd.DataFrame({"ds": future_dates})
```

## Génération des features futures
```markdown
Reproduire exactement les mêmes fonctions utilisées lors du training :

- features temporelles
- lags
- rolling windows
- variations / diff
- moyens saisonniers
- exogènes moyens / prédits

(ces fonctions sont déjà dans le script d'entraînement (voir Etape 8))
```
## 3. Prédictions
```python
# Prophet
future_prophet = future_df.copy()

for reg in model_config["Prophet"]["regressors"]:
  future_prophet[reg] = historical_df[reg].mean()

prophet_pred = prophet_model.predict(future_prophet)["yhat"].values

# LightGBM
X_future = prepare_features(future_df, historical_df)
lgb_pred = lgb_model.predict(X_future)

# SARIMAX
scaled_exog = exog_scaler.transform(future_df[model_config["SARIMAX"]["exog"]].values)
sarimax_pred = sarimax_model.get_forecast(
  steps=len(future_dates),
  exog=scaled_exog
).predicted_mean.values

# Stacking (SARIMAX + LightGBM)
import numpy as np
ensemble_input = np.column_stack([
  sarimax_pred['Y'].values, 
  lgb_pred['Y'].values
])

final_pred = stacking_model.predict(ensemble_input)
```

## 4. Évaluation (optionnel)
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_true, final_pred)
rmse = mean_squared_error(y_true, final_pred, squared=False)
print(mae, rmse)
```

## 5. Notes importantes
- Toujours utiliser les mêmes transformations de features
- Toujours recalculer les features futures à partir de l’historique.
- Réentraîner les modèles chaque mois ou trimestre.
- Garder les mêmes colonnes + même scaler.
- Ne jamais toucher l’ordre des features LightGBM.
