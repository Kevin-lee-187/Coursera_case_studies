# 🗺️ Waze User Churn Analysis

A comprehensive machine learning project to predict user churn for Waze app users using the PACE methodology (Plan, Analyze, Construct, Execute).

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![Pandas](https://img.shields.io/badge/pandas-1.3+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📊 Project Overview

This project analyzes Waze user behavior data to build predictive models that identify users at risk of churning. The analysis follows Google's PACE methodology to ensure systematic and thorough examination of the business problem.

### 🎯 Key Objectives
- Predict user churn with high accuracy (>75% AUC)
- Identify key behavioral indicators of churn risk
- Provide actionable business recommendations
- Create interpretable models for business stakeholders

### 🏆 Key Results
- **75.4% AUC** achieved with Logistic Regression model
- **17.7% overall churn rate** identified
- **Activity days** found to be the strongest predictor of churn
- **8 engineered features** that outperform raw metrics

## 📁 Project Structure

```
Waze-case-study/
├── PACE/
│   ├── waze_churn_analysis.py      # Main analysis script
│   ├── waze_dataset.csv            # Dataset (14,999 user records)
│   ├── waze_eda_visualizations.png # EDA plots
│   └── model_evaluation.png        # Model comparison charts
├── Executive_summary/
│   ├── waze_churn_analysis_executive_summary.md
│   ├── generate_html_report.py     # HTML report generator
│   ├── waze_churn_analysis_report.html
│   ├── waze_eda_visualizations.png
│   └── model_evaluation.png
├── Waze_Workflow/
│   ├── waze_churn_analysis_workflow.md
│   ├── generate_workflow_html.py
│   └── waze_churn_analysis_workflow.html
└── README.md
```

## 🔬 PACE Methodology

### **Plan** 🎯
- **Objective**: Predict user churn to improve retention strategies
- **Success Criteria**: AUC > 75%, interpretable features
- **Approach**: Supervised classification with multiple algorithms

### **Analyze** 📊
- **Dataset**: 14,999 users, 13 original features
- **Data Quality**: 700 missing labels removed (final: 14,299 records)
- **Key Insights**: Activity days strongly correlate with retention

### **Construct** 🛠️
- **Feature Engineering**: 8 new behavioral ratio features
- **Models Tested**: Logistic Regression, Random Forest, Gradient Boosting, Decision Tree
- **Preprocessing**: StandardScaler for linear models, label encoding for categorical variables

### **Execute** 🚀
- **Best Model**: Logistic Regression (75.4% AUC)
- **Cross-Validation**: 5-fold CV with consistent performance
- **Feature Importance**: Behavioral ratios outperform raw counts

## 📈 Key Findings

### Model Performance
| Model | AUC Score | Cross-Validation AUC |
|-------|-----------|---------------------|
| **Logistic Regression** | **75.4%** | **75.1% ± 2.1%** |
| Gradient Boosting | 74.7% | 74.3% ± 1.8% |
| Random Forest | 72.5% | 72.1% ± 2.3% |
| Decision Tree | 56.7% | 56.2% ± 3.1% |

### Top Predictive Features (Engineered)
1. **active_days_ratio** - Consistency of engagement over user tenure
2. **avg_drives_per_day** - Daily driving intensity and habits  
3. **avg_sessions_per_day** - Daily app engagement patterns
4. **driving_days_ratio** - Proportion of days actually driving
5. **total_nav_favs** - Navigation ecosystem investment

### Business Insights
- **No device bias**: Android (17.6%) vs iPhone (17.8%) churn rates nearly identical
- **Activity threshold**: Users with <10 activity days show highest churn risk
- **Engagement quality**: Behavioral ratios more predictive than raw usage counts
- **Ecosystem investment**: Navigation favorites usage correlates with retention

## 🛠️ Installation & Setup

### Prerequisites
```bash
python >= 3.8
pip >= 21.0
```

### Required Packages
```bash
pip install pandas numpy matplotlib seaborn scikit-learn markdown
```

### Alternative Installation
```bash
# For specific Python version (if needed)
python3.13 -m pip install pandas numpy matplotlib seaborn scikit-learn markdown
```

## 🚀 Usage

### Run Complete Analysis
```bash
cd PACE/
python waze_churn_analysis.py
```

### Generate HTML Report
```bash
cd Executive_summary/
python generate_html_report.py
```

### Expected Outputs
- `waze_eda_visualizations.png` - 9 comprehensive EDA plots
- `model_evaluation.png` - Model comparison charts
- `waze_churn_analysis_report.html` - Professional HTML report

## 📊 Visualizations

The analysis generates 9 comprehensive visualizations:

1. **Box Plot of Drives** - Distribution analysis (leadership request)
2. **Scatter Plot: Drives vs Sessions** - Usage relationship (leadership request)
3. **Churn Distribution** - Overall churn rate visualization
4. **Device Distribution by Churn** - Platform analysis
5. **Activity Days Distribution** - Key predictor analysis
6. **Driving Behavior Comparison** - Usage pattern analysis
7. **Correlation Heatmap** - Feature relationship matrix
8. **Navigation Favorites Analysis** - Ecosystem engagement
9. **Days After Onboarding** - User lifecycle analysis

## 💼 Business Recommendations

### 🎯 Immediate Actions
1. **Target High-Risk Users**: Focus on users with <10 activity days
2. **Engagement Campaigns**: Implement behavioral trigger-based outreach
3. **Onboarding Optimization**: Improve early user experience

### 📱 Product Development
1. **Promote Navigation Favorites**: Increase ecosystem investment
2. **Daily Engagement Features**: Focus on consistent daily usage
3. **Quality over Quantity**: Emphasize meaningful interactions

### 📊 Implementation Strategy
1. **Real-time Scoring**: Deploy Logistic Regression model for live predictions
2. **A/B Testing**: Test retention strategies on identified high-risk segments
3. **Regular Model Updates**: Monthly retraining with new user data

## 🔄 Model Deployment

### Recommended Approach
- **Model**: Logistic Regression (interpretable and high-performing)
- **Scoring Frequency**: Real-time or daily batch processing
- **Threshold**: Probability > 0.5 for churn prediction
- **Update Schedule**: Monthly retraining recommended

### Risk Segmentation
- **Low Risk**: active_days_ratio > 0.6, total_nav_favs > 10
- **Medium Risk**: active_days_ratio 0.3-0.6, moderate engagement
- **High Risk**: active_days_ratio < 0.3, declining patterns

## 📋 File Descriptions

| File | Description |
|------|-------------|
| `waze_churn_analysis.py` | Main analysis script with PACE methodology |
| `waze_dataset.csv` | Original dataset (14,999 user records) |
| `generate_html_report.py` | Professional HTML report generator |
| `waze_churn_analysis_executive_summary.md` | Business-focused summary |
| `waze_churn_analysis_workflow.md` | Detailed methodology walkthrough |
| `*.png` | Generated visualizations and model evaluation charts |
| `*.html` | Professional web-ready reports |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PACE Methodology**: Google Advanced Data Analytics Certificate framework
- **Business Stakeholders**: Leadership team for specific visualization requests
- **Data Science Community**: Open-source tools and methodologies

## 📞 Contact

For questions about this analysis or potential collaboration:

- **Project Repository**: [GitHub Repository URL]
- **Analysis Date**: 2024
- **Model Performance**: 75.4% AUC (exceeds 75% threshold)

---

**Note**: This project demonstrates end-to-end machine learning workflow from business problem definition to actionable insights and deployment recommendations. 