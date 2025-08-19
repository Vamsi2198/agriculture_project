# TerraCred™: AI-Powered Credit Scoring for Smallholder Farmers

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Enabled-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-orange.svg)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **End-to-End Predictive Credit Scoring System leveraging satellite data, weather patterns, and socio-economic indicators to provide instant, explainable credit recommendations for smallholder farmers in India.**

## 🌱 Problem Statement

Smallholder farmers in India face significant barriers to accessing timely credit:
- **Lengthy paperwork** and collateral requirements
- **Outdated credit bureau scores** ignoring agricultural realities
- **Missed optimal timing** for Kisan Credit Card (KCC) top-ups
- **Limited consideration** of current crop health and weather risks

## 🎯 Solution: TerraCred™

TerraCred™ revolutionizes agricultural credit assessment by:

- 🛰️ **Leveraging zero-cost public datasets** (NDVI, IMD rainfall, Soil Health Cards)
- 🤖 **AI-powered instant predictions** with explainable results
- 💻 **Low-cost deployment** on ₹2,500 Raspberry Pi Zero (in loop)
- 🌍 **Multi-language support** via SMS, WhatsApp, or chatbot (in loop)
- ⚡ **Dual prediction models**: Credit limit estimation + loan approval classification


Generated SHAP visualizations:
📊 Bar plots (Feature Importance)
<img width="1264" height="1504" alt="shap_reg_bar" src="https://github.com/user-attachments/assets/b0a486ce-6c43-43f1-9051-fc69c045b175" />

🐝 Beeswarm plots (Positive/Negative Impact)  
<img width="1246" height="1504" alt="shap_reg_beeswarm" src="https://github.com/user-attachments/assets/f8a1141c-4bee-4afc-a684-ee71963d8264" />

⚡ Feature Importance plots
<img width="1280" height="1280" alt="permutation_importance_top25" src="https://github.com/user-attachments/assets/01821316-f587-4062-9102-c1cca1558219" />



## 📊 Model Architecture

### Dual Prediction System

| Model Type | Target Variable | Purpose | Metrics |
|------------|----------------|---------|---------|
| **Regression** | `credit_limit` | Predict optimal credit amount (₹0-100K) | RMSE, MAE, R² |
| **Classification** | `loan_approval` | Binary approval recommendation | Accuracy, ROC-AUC |

### Algorithm Choice: LightGBM

- ✅ **Heterogeneous data handling** (continuous, categorical, binary)
- ✅ **Efficient performance** with 4000+ sample dataset
- ✅ **Non-linear interaction capture** without manual feature engineering
- ✅ **Native SHAP compatibility** for explainability
- 🔄 **Fallback**: RandomForest if LightGBM unavailable

## 🔧 Feature Engineering

### 34 Engineered Features Across 5 Categories:

#### 🌿 **Environmental Indicators**
- `ndvi_max`, `ndvi_trend_30d` - Crop health & growth trends
- `rain_7d`, `rain_forecast_14d`, `dry_spell` - Water availability
- `soil_N`, `soil_K`, `soil_pH` - Soil fertility metrics

#### 💰 **Market & Economic**
- `price_volatility`, `avg_crop_price_last_3m` - Price dynamics
- `market_access_score`, `distance_to_market_km` - Market connectivity

#### 🏛️ **Socio-Economic**
- `pmkisan_beneficiary`, `fpo_membership` - Government support
- `fertilizer_subsidy`, `crop_insurance` - Safety nets
- `mobile_penetration_score` - Digital connectivity

#### ⚠️ **Risk Factors**
- `past_default`, `past_repayment_rate` - Credit history
- `yield_variability_3y` - Production stability

#### 🚜 **Farm Characteristics**
- `plot_size_acres`, `irrigation`, `crop_type` - Farm infrastructure

### Advanced Transformations

```python
# Risk-adjusted water stress
water_stress_index = dry_spell / (rain_7d + 1.0)

# Market advantage composite
market_advantage = market_access_score - log_distance_to_market

# Growth outlook integration  
growth_outlook = (ndvi_trend_30d * 5000 + rain_forecast_14d)

# Digital enablement score
digital_enablement = 0.6 * mobile_penetration + 0.4 * market_access
```

## 📈 Model Performance

### Regression (Credit Limit Prediction)
- **RMSE**: ~1,200 (on ₹0-100K scale)
- **R²**: 0.85+ 
- **MAE**: <1,000

### Classification (Loan Approval)
- **Accuracy**: 89%+
- **ROC-AUC**: 0.92+


### Credit Limit Formula

```python
credit_limit = (
    Base_Capacity +          # NDVI, soil nutrients, irrigation
    Seasonal_Factors +       # Rainfall, growth trends
    Market_Factors +         # Prices, volatility, access
    Social_Support +         # Govt aid, subsidies, insurance  
    Size_Effects +           # Plot size scaling
    Risk_Penalties +         # Defaults, repayment history
    Random_Noise             # Natural variation
).clip(0, 100000)
```

### Loan Approval Logic

```python
approval_score = (
    0.00006 * credit_limit +
    2.5 * past_repayment_rate -
    2.0 * past_default -
    3.0 * yield_variability_3y +
    0.08 * market_access_score +
    noise
)
loan_approval = sigmoid(approval_score) > 0.5
```


## 📋 Requirements

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.3.0
joblib>=1.0.0
lightgbm>=3.2.0  # Optional but recommended
shap>=0.40.0     # For explainability
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## 📊 Sample Predictions

### High Credit Limit Farmer Profile
```json
{
  "credit_limit": 85000,
  "loan_approval": 1,
  "top_positive_factors": [
    "High NDVI (healthy crops)",
    "Excellent repayment history", 
    "Irrigation access",
    "PM-Kisan beneficiary"
  ],
  "risk_factors": [
    "High price volatility",
    "Distance to market"
  ]
}
```

### Low Credit Limit Farmer Profile
```json
{
  "credit_limit": 12000,
  "loan_approval": 0,  
  "limiting_factors": [
    "Past loan default",
    "Poor soil nutrients",
    "High yield variability",
    "No irrigation access"
  ],
  "improvement_suggestions": [
    "Soil testing and fertilization",
    "Crop insurance enrollment", 
    "Join local FPO"
  ]
}
```

## 🌍 Impact & Vision

### Immediate Benefits
- ⚡ **60-second credit assessment** vs weeks of traditional processing
- 🎯 **90%+ accuracy** with transparent, explainable decisions
- 💰 **Zero-cost data sources** making system sustainable
- 📱 **Multi-language accessibility** for rural populations

### Scale Potential
- 🏦 **Bank partnerships**: Integration with existing KCC systems
- 🌾 **Crop-specific models**: Specialized for rice, wheat, cotton, etc.
- 📡 **Real-time updates**: Integration with satellite data APIs
- 🤝 **Government adoption**: Support for PM-Kisan and DBT schemes

---

**Built with ❤️ for India's farming community**

*Empowering 140+ million farmers with AI-driven financial inclusion*
