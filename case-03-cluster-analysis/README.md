# MegaMart Customer Segmentation Analysis


## 👥 Team

| Name | Student ID |
|------|------------|
| César Isao Pastelin Kohagura | A01659947 |
| Luis Emilio Fernández González | A01659517 |
| Eduardo Botello Casey | A01659281 |

---

## 📋 Project Overview

This project implements a comprehensive customer segmentation analysis for **MegaMart**, a retail company facing strategic challenges with generic marketing campaigns that fail to account for customer behavior diversity. The analysis uses clustering techniques to identify distinct customer profiles and translate them into actionable business strategies.

### Business Problem

MegaMart's current challenges include:
- Generic marketing campaigns with low engagement
- Inefficient resource allocation
- Poor performance on key metrics (CLV, MROI, Churn)
- Inability to detect high-risk customers early for retention actions

### Solution Approach

Data-driven segmentation using:
- **Hierarchical Clustering** (Ward's linkage)
- **K-Means Clustering**
- **PCA** for dimensionality reduction and visualization

---

## 📊 Dataset Description

| Attribute | Value |
|-----------|-------|
| **Total Customers** | 3,000 |
| **Variables Analyzed** | 9 |
| **Missing Values** | None |

### Variables

| Variable | Description | Type |
|----------|-------------|------|
| `monthly_transactions` | Number of transactions per month | float64 |
| `avg_basket_size` | Average items per purchase | float64 |
| `total_spend` | Total spending amount | float64 |
| `avg_session_duration` | Average browsing session length | float64 |
| `email_open_rate` | Rate of opened marketing emails | float64 |
| `product_views_per_visit` | Products viewed per session | float64 |
| `return_rate` | Product return percentage | float64 |
| `customer_tenure_months` | Duration as customer | int64 |
| `recency_days` | Days since last purchase | int64 |

### Key Correlations

| Variable Pair | Correlation |
|--------------|-------------|
| avg_basket_size ↔ total_spend | 0.941 |
| monthly_transactions ↔ total_spend | 0.764 |
| monthly_transactions ↔ avg_basket_size | 0.691 |
| monthly_transactions ↔ recency_days | -0.632 |
| total_spend ↔ recency_days | -0.612 |

---

## 🔬 Methodology

### 1. Data Preprocessing
- Removed `customer_id` (non-informative)
- Applied **StandardScaler** for normalization
- No missing data imputation needed

### 2. Clustering Approach

#### Hierarchical Clustering
- Tested linkage methods: Single, Complete, Average, **Ward** (selected)
- Ward's method chosen for minimizing within-cluster variance

#### K-Means Clustering
- Evaluated k = 2 to 10 clusters
- Used **Elbow Method** and **Silhouette Analysis**

### 3. Optimal Cluster Selection

| K | Silhouette Score | Min Size | Max Size |
|---|------------------|----------|----------|
| 2 | 0.344 | 540 | 2,460 |
| 3 | 0.295 | 540 | 1,501 |
| **4** | **0.316** | **420** | **1,081** |
| 5 | 0.300 | 420 | 1,081 |
| 6 | 0.248 | 373 | 708 |

**Selected: k = 4** (balance between statistical quality and business interpretability)

### 4. Validation
- Silhouette Score (K-Means): **0.317**
- Silhouette Score (Hierarchical): **0.316**
- PCA variance explained (2D): 62.0%
- PCA variance explained (3D): 74.6%

---

## 📈 Customer Segments

### Cluster 0 — High-Value Champions (17.5%)
**525 customers**

| Metric | Value | vs. Average |
|--------|-------|-------------|
| Monthly Transactions | 14.07 | +134.1% |
| Avg Basket Size | 22.03 | +132.2% |
| Total Spend | $6,507 | — |
| Recency Days | 8.02 | -61.2% |
| Return Rate | 0.10 | -46.7% |
| Tenure | 26.2 months | +46.4% |

**Profile:** Highly profitable customers with high purchase frequency, large basket sizes, low recency, and long tenure.

**Strategic Objective:** Prioritized retention and value expansion.

---

### Cluster 1 — Window Shoppers (31.0%)
**929 customers**

| Metric | Value | vs. Average |
|--------|-------|-------------|
| Monthly Transactions | 1.68 | -72.0% |
| Avg Basket Size | 3.05 | -67.8% |
| Total Spend | $423 | — |
| Recency Days | 35.59 | +72.1% |
| Return Rate | 0.27 | +47.8% |
| Session Duration | 52.31 | +36.1% |

**Profile:** Users who browse frequently but rarely convert, exhibit high recency, and show elevated return rates.

**Strategic Objective:** Reactivation and conversion.

---

### Cluster 2 — Premium Occasional Buyers (14.4%)
**433 customers**

| Metric | Value | vs. Average |
|--------|-------|-------------|
| Monthly Transactions | 4.04 | -32.8% |
| Avg Basket Size | 18.17 | +91.6% |
| Total Spend | $3,876 | — |
| Session Duration | 22.36 | -41.8% |
| Product Views/Visit | 16.55 | -47.2% |
| Return Rate | 0.24 | +31.7% |

**Profile:** Customers who make few but high-value purchases; short sessions with minimal browsing.

**Strategic Objective:** Increase purchase frequency and foster recurrence.

---

### Cluster 3 — Low-Engagement Mass Segment (37.1%)
**1,113 customers**

| Metric | Value | vs. Average |
|--------|-------|-------------|
| Monthly Transactions | 6.59 | ~average |
| Avg Basket Size | 5.56 | -41.4% |
| Total Spend | $1,451 | — |
| Recency Days | 14.53 | -29.7% |
| Return Rate | 0.13 | -30.1% |

**Profile:** The largest but least profitable segment: low ticket size, low interaction, and limited contribution to revenue.

**Strategic Objective:** Improve profitability through cost-effective approaches.

---

## 💼 Strategic Recommendations

### Cluster 0 — High-Value Champions
- ✅ Premium loyalty program with early access and exclusive services
- ✅ Personalized campaigns using model-driven recommendations
- ✅ VIP churn alerts with automated triggers
- ✅ Priority customer support

### Cluster 1 — Window Shoppers
- ✅ Dynamic retargeting with product reminders
- ✅ Optimized checkout flow (reduce friction)
- ✅ Reactivation incentives ("welcome back" discounts)
- ✅ Recency-based email journeys

### Cluster 2 — Premium Occasional Buyers
- ✅ Complementary product recommendations
- ✅ Subscriptions or scheduled reminders
- ✅ Simplified customer journey
- ✅ Second-purchase incentives

### Cluster 3 — Low-Engagement Mass Segment
- ✅ Volume promotions and bundles
- ✅ Mass automation for communications
- ✅ Inventory optimization for high-rotation items
- ✅ Upward-mobility detection

---

## 📅 Implementation Roadmap

### Q1 — Foundations and Initial Activation
- Deploy recency-based automation journeys (Clusters 1 & 2)
- Activate dynamic retargeting
- Design premium loyalty program (Cluster 0)
- Optimize checkout flow (Cluster 1)
- Launch volume promotions (Cluster 3)

### Q2 — Personalization and Smart Retention
- Deploy model-based personalized recommendations
- Implement VIP churn alert system
- Enable fast-purchase journeys (Cluster 2)
- Automate mass communications (Cluster 3)

### Q3 — Value Expansion
- Launch full premium loyalty program
- Implement second-purchase incentive campaigns
- Execute advanced reactivation campaigns
- Align inventory with usage patterns

### Q4 — Optimization and Scale
- Conduct full impact assessment
- Retrain clustering model
- Promote high-potential Cluster 3 users
- Refine marketing based on ROI

---

## 📊 Expected Business Impact

| Cluster | Expected Outcome |
|---------|------------------|
| **Cluster 0** | Maintain low churn, high frequency, strong basket size |
| **Cluster 1** | 40% reduction in return rate, 50% increase in transactions |
| **Cluster 2** | 40% increase in monthly visits, 30% growth in purchases |
| **Cluster 3** | 30% increase in basket size, 30% improvement in engagement |

---

## 🛠️ Technical Requirements

```python
# Required Libraries
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
```

### Key Functions Used
- `StandardScaler` - Data normalization
- `linkage`, `dendrogram`, `fcluster` - Hierarchical clustering
- `KMeans` - K-Means clustering
- `silhouette_score`, `silhouette_samples` - Cluster validation
- `PCA` - Dimensionality reduction

---

## 📁 Project Structure

```
├── Team3_Analysis.pdf          # Full Jupyter notebook with code and analysis
├── Team3_Slides.pdf            # Business presentation slides
├── Team3_Summary.pdf           # Executive summary report
├── retail_customer_data-1.csv  # Source dataset
└── README.md                   # This file
```

---

## 📹 Presentation Video

🎥 [Watch on YouTube](https://youtu.be/1ee-Rb251Y4)


## 📄 License

This project was developed for academic purposes as part of the course "Aplicación de Métodos Multivariados en Ciencia de Datos" at Tecnológico de Monterrey.

---

*Last updated: November 2025*

