# Système de Prévision des Ventes - Supermarché

## Vue d'ensemble
Système complet de prévision des ventes quotidiennes utilisant:
- Modèles statistiques (Prophet, SARIMAX)
- Machine Learning (LightGBM)
- Deep Learning (LSTM)
- Ensembling Ridge pour combiner les modeles

## Structure
- `prevision_des_ventes_supermarket.ipynb`: Notebook principal avec tous les modèles
- `scaler/exog_scaler.pkl`: Scalers pour exogenes SARIMAX
- `metadata/`: Configurations et infos d'entraînement
- `models/`: Modèles sauvegardés
- `USAGE_GUIDE.md`: Guide de réutilisation

## Installation
```bash
pip install -r requirements.txt
```

## Utilisation rapide
```python
# Voir USAGE_GUIDE.md pour guide détaillé
from prophet import Prophet
import pickle
import pandas as pd

# Charger modèles
with open('preprocessing_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)
prophet_model = Prophet.load('prophet_full_model')

# Prédire 7 jours
future = pd.DataFrame({'ds': pd.date_range(start='2025-01-13', periods=7, freq='D')})
forecast = prophet_model.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
```

## Notes importantes
1. Réentraîner hebdomadairement avec nouvelles données
2. Monitorer erreurs vs actuel pour détecter drift
3. Pour meilleure précision: segmenter par branche
4. Vérifier hypothèses SARIMAX avant utilisation

## Support
Pour questions/bugs, voir notebook ou USAGE_GUIDE.md
