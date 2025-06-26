#!/usr/bin/env python3
"""
Waze User Churn Analysis
Following PACE Strategy: Plan, Analyze, Construct, Execute

This analysis predicts user churn for Waze app users using machine learning techniques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Load and perform initial exploration of the Waze dataset"""
    print("=" * 60)
    print("PACE STRATEGY - PLAN PHASE")
    print("=" * 60)
    print("Objective: Predict user churn for Waze app users")
    print("Approach: Follow PACE methodology")
    print("- Plan: Define objectives and approach")
    print("- Analyze: EDA and data understanding")
    print("- Construct: Build ML models")
    print("- Execute: Evaluate and present results")
    print()
    
    # Load the dataset
    df = pd.read_csv('PACE/waze_dataset.csv')
    
    print("=" * 60)
    print("PACE STRATEGY - ANALYZE PHASE")
    print("=" * 60)
    print("1. INITIAL DATA EXPLORATION")
    print("=" * 30)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    print("First 5 rows:")
    print(df.head())
    print()
    
    print("Data types:")
    print(df.dtypes)
    print()
    
    print("Basic statistics:")
    print(df.describe())
    print()
    
    return df

def analyze_data_quality(df):
    """Analyze data quality and identify issues"""
    print("2. DATA QUALITY ANALYSIS")
    print("=" * 30)
    
    # Check for missing values
    print("Missing values:")
    missing_data = df.isnull().sum()
    print(missing_data[missing_data > 0])
    print()
    
    # Check for empty strings in label column
    print("Unique values in 'label' column:")
    print(df['label'].value_counts(dropna=False))
    print()
    
    # Check for duplicates
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print()
    
    # Check for outliers in numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print("Numerical columns statistics:")
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")
    print()

def clean_data(df):
    """Clean the dataset"""
    print("3. DATA CLEANING")
    print("=" * 30)
    
    df_clean = df.copy()
    
    # Handle missing values in label column (empty strings)
    print(f"Rows with empty label before cleaning: {(df_clean['label'] == '').sum()}")
    
    # Remove rows with empty or missing labels
    df_clean = df_clean[df_clean['label'].notna()]
    df_clean = df_clean[df_clean['label'] != '']
    
    print(f"Rows with empty label after cleaning: {(df_clean['label'] == '').sum()}")
    print(f"Dataset shape after cleaning: {df_clean.shape}")
    print()
    
    # Handle missing values in other columns
    print("Handling missing values in numerical columns:")
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"Filled {col} missing values with median: {median_val}")
    
    print("Final dataset info:")
    print(df_clean.info())
    print()
    
    return df_clean

def create_visualizations(df):
    """Create data visualizations as requested"""
    print("4. DATA VISUALIZATION")
    print("=" * 30)
    
    # Define consistent color palette for better differentiation
    churn_colors = {'retained': '#1f77b4', 'churned': '#ff7f0e'}  # Blue and orange
    device_colors = ['#4472C4', '#E70000']  # Blue and red
    hist_colors = ['#2E8B57', '#DC143C']  # Matching churn colors
    
    # Set up the plotting area with larger size and better spacing
    fig = plt.figure(figsize=(24, 18))
    plt.rcParams.update({'font.size': 10})
    
    # 1. Box plot of drives (as specifically requested)
    plt.subplot(3, 3, 1)
    box_plot = sns.boxplot(data=df, y='drives', color='#4472C4', width=0.6)
    plt.title('Box Plot of Drives', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Number of Drives', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 2. Scatter plot of drives vs sessions (as specifically requested)
    plt.subplot(3, 3, 2)
    for i, label in enumerate(df['label'].unique()):
        subset = df[df['label'] == label]
        plt.scatter(subset['drives'], subset['sessions'], 
                   alpha=0.7, label=label, color=churn_colors.get(label, 'gray'), 
                   s=30, edgecolors='white', linewidth=0.5)
    plt.xlabel('Number of Drives', fontsize=12)
    plt.ylabel('Number of Sessions', fontsize=12)
    plt.title('Scatter Plot: Drives vs Sessions', fontsize=16, fontweight='bold', pad=20)
    plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 3. Churn distribution
    plt.subplot(3, 3, 3)
    churn_counts = df['label'].value_counts()
    colors_pie = [churn_colors[label] for label in churn_counts.index]
    wedges, texts, autotexts = plt.pie(churn_counts.values, labels=churn_counts.index, 
                                      autopct='%1.1f%%', colors=colors_pie, 
                                      startangle=90, textprops={'fontsize': 11})
    plt.title('Distribution of User Labels', fontsize=16, fontweight='bold', pad=20)
    
    # 4. Device distribution by churn
    plt.subplot(3, 3, 4)
    device_churn = pd.crosstab(df['device'], df['label'])
    ax = device_churn.plot(kind='bar', ax=plt.gca(), 
                          color=[churn_colors['churned'], churn_colors['retained']], 
                          width=0.7)
    plt.title('Device Type by Churn Status', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Device Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0, fontsize=11)
    plt.legend(title='Status', frameon=True, fancybox=True, shadow=True, fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 5. Activity days distribution
    plt.subplot(3, 3, 5)
    sns.histplot(data=df, x='activity_days', hue='label', bins=25, alpha=0.8,
                palette=churn_colors, kde=True)
    plt.title('Distribution of Activity Days', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Activity Days', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Status', frameon=True, fancybox=True, shadow=True, fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 6. Driving behavior comparison
    plt.subplot(3, 3, 6)
    df_melted = df.melt(id_vars=['label'], 
                       value_vars=['driven_km_drives', 'duration_minutes_drives'],
                       var_name='metric', value_name='value')
    # Rename metrics for better readability
    metric_labels = {'driven_km_drives': 'Distance (km)', 'duration_minutes_drives': 'Duration (min)'}
    df_melted['metric'] = df_melted['metric'].map(metric_labels)
    
    sns.boxplot(data=df_melted, x='metric', y='value', hue='label', palette=churn_colors)
    plt.title('Driving Behavior Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.xticks(fontsize=11)
    plt.legend(title='Status', frameon=True, fancybox=True, shadow=True, fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 7. Correlation heatmap
    plt.subplot(3, 3, 7)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='RdBu_r', center=0, 
                square=True, cbar_kws={'shrink': 0.8}, 
                xticklabels=True, yticklabels=True)
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    # 8. Navigation favorites analysis
    plt.subplot(3, 3, 8)
    df['total_nav_favs'] = df['total_navigations_fav1'] + df['total_navigations_fav2']
    sns.boxplot(data=df, x='label', y='total_nav_favs', palette=churn_colors, width=0.6)
    plt.title('Total Navigation Favorites by Churn', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('User Status', fontsize=12)
    plt.ylabel('Total Navigation Favorites', fontsize=12)
    plt.xticks(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 9. Days after onboarding
    plt.subplot(3, 3, 9)
    sns.histplot(data=df, x='n_days_after_onboarding', hue='label', bins=40, alpha=0.8,
                palette=churn_colors, kde=True)
    plt.title('Days After Onboarding Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Days After Onboarding', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Status', frameon=True, fancybox=True, shadow=True, fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Adjust layout with better spacing
    plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)
    plt.savefig('waze_eda_visualizations.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("Visualizations created and saved as 'waze_eda_visualizations.png'")
    print()

def perform_statistical_analysis(df):
    """Perform statistical analysis"""
    print("5. STATISTICAL ANALYSIS")
    print("=" * 30)
    
    # Basic statistics by churn status
    print("Key metrics by churn status:")
    grouped_stats = df.groupby('label').agg({
        'sessions': ['mean', 'median', 'std'],
        'drives': ['mean', 'median', 'std'],
        'activity_days': ['mean', 'median', 'std'],
        'driving_days': ['mean', 'median', 'std'],
        'driven_km_drives': ['mean', 'median', 'std'],
        'duration_minutes_drives': ['mean', 'median', 'std']
    }).round(2)
    
    print(grouped_stats)
    print()
    
    # Calculate churn rate
    churn_rate = df['label'].value_counts(normalize=True)
    print("Churn rates:")
    print(churn_rate)
    print()
    
    # Device analysis
    print("Device analysis:")
    device_analysis = pd.crosstab(df['device'], df['label'], normalize='index')
    print(device_analysis)
    print()

def prepare_features(df):
    """Prepare features for machine learning"""
    print("=" * 60)
    print("PACE STRATEGY - CONSTRUCT PHASE")
    print("=" * 60)
    print("1. FEATURE ENGINEERING")
    print("=" * 30)
    
    df_ml = df.copy()
    
    # Create new features
    df_ml['avg_sessions_per_day'] = df_ml['sessions'] / (df_ml['activity_days'] + 1)  # +1 to avoid division by zero
    df_ml['avg_drives_per_day'] = df_ml['drives'] / (df_ml['driving_days'] + 1)
    df_ml['avg_km_per_drive'] = df_ml['driven_km_drives'] / (df_ml['drives'] + 1)
    df_ml['avg_duration_per_drive'] = df_ml['duration_minutes_drives'] / (df_ml['drives'] + 1)
    df_ml['sessions_drives_ratio'] = df_ml['sessions'] / (df_ml['drives'] + 1)
    df_ml['total_nav_favs'] = df_ml['total_navigations_fav1'] + df_ml['total_navigations_fav2']
    df_ml['active_days_ratio'] = df_ml['activity_days'] / (df_ml['n_days_after_onboarding'] + 1)
    df_ml['driving_days_ratio'] = df_ml['driving_days'] / (df_ml['n_days_after_onboarding'] + 1)
    
    # Handle infinite values
    df_ml.replace([np.inf, -np.inf], 0, inplace=True)
    
    print("New features created:")
    new_features = ['avg_sessions_per_day', 'avg_drives_per_day', 'avg_km_per_drive', 
                   'avg_duration_per_drive', 'sessions_drives_ratio', 'total_nav_favs',
                   'active_days_ratio', 'driving_days_ratio']
    for feature in new_features:
        print(f"- {feature}")
    print()
    
    return df_ml

def build_ml_models(df):
    """Build and evaluate machine learning models"""
    print("2. MACHINE LEARNING MODEL CONSTRUCTION")
    print("=" * 30)
    
    # Prepare features and target
    # Exclude non-predictive columns
    exclude_cols = ['ID', 'label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df['label'].copy()
    
    # Encode categorical variables
    le_device = LabelEncoder()
    X['device_encoded'] = le_device.fit_transform(X['device'])
    X = X.drop('device', axis=1)
    
    # Encode target variable
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    print(f"Features: {X.columns.tolist()}")
    print(f"Target classes: {le_target.classes_}")
    print(f"Feature matrix shape: {X.shape}")
    print()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print()
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    # Train and evaluate models
    model_results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Use scaled data for Logistic Regression, original for tree-based models
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        if name == 'Logistic Regression':
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        
        model_results[name] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'auc_score': auc_score,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"AUC Score: {auc_score:.4f}")
        print(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print()
    
    return model_results, X_test, y_test, le_target, feature_cols

def evaluate_models(model_results, X_test, y_test, le_target):
    """Evaluate and compare models"""
    print("=" * 60)
    print("PACE STRATEGY - EXECUTE PHASE")
    print("=" * 60)
    print("1. MODEL EVALUATION")
    print("=" * 30)
    
    # Create evaluation plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC Curves
    ax1 = axes[0, 0]
    for name, results in model_results.items():
        fpr, tpr, _ = roc_curve(y_test, results['probabilities'])
        ax1.plot(fpr, tpr, label=f"{name} (AUC = {results['auc_score']:.3f})")
    ax1.plot([0, 1], [0, 1], 'k--', label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Model comparison
    ax2 = axes[0, 1]
    model_names = list(model_results.keys())
    auc_scores = [results['auc_score'] for results in model_results.values()]
    cv_means = [results['cv_mean'] for results in model_results.values()]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax2.bar(x - width/2, auc_scores, width, label='Test AUC', alpha=0.8)
    ax2.bar(x + width/2, cv_means, width, label='CV AUC', alpha=0.8)
    ax2.set_xlabel('Models')
    ax2.set_ylabel('AUC Score')
    ax2.set_title('Model Performance Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Best model confusion matrix
    best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['auc_score'])
    best_predictions = model_results[best_model_name]['predictions']
    
    ax3 = axes[1, 0]
    cm = confusion_matrix(y_test, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_title(f'Confusion Matrix - {best_model_name}')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    ax3.set_xticklabels(le_target.classes_)
    ax3.set_yticklabels(le_target.classes_)
    
    # Feature importance focusing on new engineered features
    if 'Random Forest' in model_results:
        ax4 = axes[1, 1]
        rf_model = model_results['Random Forest']['model']
        
        # Define the new engineered features
        new_features = [
            'avg_sessions_per_day', 
            'avg_drives_per_day', 
            'avg_km_per_drive',
            'avg_duration_per_drive', 
            'sessions_drives_ratio', 
            'total_nav_favs',
            'active_days_ratio', 
            'driving_days_ratio'
        ]
        
        # Get all feature importances
        all_feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': rf_model.feature_importances_
        })
        
        # Filter for new features and add a few key original features for context
        key_original_features = ['activity_days', 'sessions', 'drives']
        features_to_show = new_features + [f for f in key_original_features if f in X_test.columns]
        
        feature_importance = all_feature_importance[
            all_feature_importance['feature'].isin(features_to_show)
        ].sort_values('importance', ascending=False)
        
        # Create color coding: new features in blue, original in orange
        colors = ['#1f77b4' if feat in new_features else '#ff7f0e' 
                 for feat in feature_importance['feature']]
        
        sns.barplot(data=feature_importance, x='importance', y='feature', 
                   palette=colors, ax=ax4)
        ax4.set_title('Feature Importance: Engineered vs Original Features')
        ax4.set_xlabel('Importance Score')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#1f77b4', label='Engineered Features'),
                          Patch(facecolor='#ff7f0e', label='Original Features')]
        ax4.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed results
    print("DETAILED MODEL RESULTS:")
    print("=" * 30)
    
    for name, results in model_results.items():
        print(f"\n{name}:")
        print("-" * 40)
        print(f"Test AUC Score: {results['auc_score']:.4f}")
        print(f"CV AUC Score: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})")
        print("\nClassification Report:")
        print(classification_report(y_test, results['predictions'], 
                                  target_names=le_target.classes_))
    
    # Identify best model
    best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['auc_score'])
    print(f"\nBEST MODEL: {best_model_name}")
    print(f"AUC Score: {model_results[best_model_name]['auc_score']:.4f}")
    
    return best_model_name

def generate_insights(df, model_results, best_model_name):
    """Generate business insights"""
    print("\n2. BUSINESS INSIGHTS")
    print("=" * 30)
    
    insights = []
    
    # Churn rate insight
    churn_rate = (df['label'] == 'churned').mean()
    insights.append(f"Overall churn rate: {churn_rate:.1%}")
    
    # Device insight
    device_churn = df.groupby('device')['label'].apply(lambda x: (x == 'churned').mean())
    insights.append(f"Device churn rates: {device_churn.to_dict()}")
    
    # Activity insight
    churned_users = df[df['label'] == 'churned']
    retained_users = df[df['label'] == 'retained']
    
    avg_activity_churned = churned_users['activity_days'].mean()
    avg_activity_retained = retained_users['activity_days'].mean()
    insights.append(f"Average activity days - Churned: {avg_activity_churned:.1f}, Retained: {avg_activity_retained:.1f}")
    
    # Driving behavior insight
    avg_drives_churned = churned_users['drives'].mean()
    avg_drives_retained = retained_users['drives'].mean()
    insights.append(f"Average drives - Churned: {avg_drives_churned:.1f}, Retained: {avg_drives_retained:.1f}")
    
    # Model performance insight
    best_auc = model_results[best_model_name]['auc_score']
    insights.append(f"Best model ({best_model_name}) achieves {best_auc:.1%} AUC score")
    
    # Feature importance insights (focus on engineered features)
    if 'Random Forest' in model_results:
        rf_model = model_results['Random Forest']['model']
        
        # Define the new engineered features
        new_features = [
            'avg_sessions_per_day', 'avg_drives_per_day', 'avg_km_per_drive',
            'avg_duration_per_drive', 'sessions_drives_ratio', 'total_nav_favs',
            'active_days_ratio', 'driving_days_ratio'
        ]
        
        # Get feature importance for engineered features
        feature_names = list(rf_model.feature_names_in_) if hasattr(rf_model, 'feature_names_in_') else list(range(len(rf_model.feature_importances_)))
        all_importance = dict(zip(feature_names, rf_model.feature_importances_))
        
        # Filter for engineered features only
        engineered_importance = {feat: all_importance.get(feat, 0) 
                               for feat in new_features if feat in all_importance}
        
        # Sort by importance
        top_engineered = sorted(engineered_importance.items(), 
                              key=lambda x: x[1], reverse=True)[:5]
        
        print("\nTOP 5 ENGINEERED FEATURE IMPORTANCE:")
        print("=" * 40)
        for i, (feature, importance) in enumerate(top_engineered, 1):
            print(f"{i}. {feature}: {importance:.4f}")
        
        insights.append(f"Top engineered feature: {top_engineered[0][0]} (importance: {top_engineered[0][1]:.4f})")
    
    print("\nKEY INSIGHTS:")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    return insights

def main():
    """Main execution function"""
    print("WAZE USER CHURN ANALYSIS")
    print("=" * 60)
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Analyze data quality
    analyze_data_quality(df)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Create visualizations
    create_visualizations(df_clean)
    
    # Perform statistical analysis
    perform_statistical_analysis(df_clean)
    
    # Prepare features
    df_ml = prepare_features(df_clean)
    
    # Build ML models
    model_results, X_test, y_test, le_target, feature_cols = build_ml_models(df_ml)
    
    # Evaluate models
    best_model_name = evaluate_models(model_results, X_test, y_test, le_target)
    
    # Generate insights
    insights = generate_insights(df_clean, model_results, best_model_name)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("Files generated:")
    print("- waze_eda_visualizations.png")
    print("- model_evaluation.png")
    print("=" * 60)
    
    return df_clean, model_results, insights

if __name__ == "__main__":
    df, models, insights = main()
