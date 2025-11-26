"""
Exploratory Data Analysis for Song Rating Prediction
This script analyzes which features are most important for predicting song ratings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("EXPLORATORY DATA ANALYSIS: Song Rating Prediction")
print("="*80)

# Load data
print("\n1. LOADING DATA...")
df = pd.read_csv('PDMX/PDMX.csv')
print(f"Dataset shape: {df.shape}")
print(f"Total songs: {len(df):,}")

# Focus on rated songs only
df_rated = df[df['is_rated'] == True].copy()
print(f"Songs with ratings: {len(df_rated):,} ({len(df_rated)/len(df)*100:.1f}%)")

print("\n" + "="*80)
print("2. RATING DISTRIBUTION ANALYSIS")
print("="*80)

# Rating statistics
print(f"\nRating Statistics:")
print(f"  Mean:   {df_rated['rating'].mean():.2f}")
print(f"  Median: {df_rated['rating'].median():.2f}")
print(f"  Std:    {df_rated['rating'].std():.2f}")
print(f"  Min:    {df_rated['rating'].min():.2f}")
print(f"  Max:    {df_rated['rating'].max():.2f}")

# Rating distribution
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.hist(df_rated['rating'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Song Ratings')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(df_rated['rating'])
plt.ylabel('Rating')
plt.title('Rating Distribution (Box Plot)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rating_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: rating_distribution.png")
plt.close()

print("\n" + "="*80)
print("3. FEATURE ENGINEERING & SELECTION")
print("="*80)

# Identify numeric features (excluding identifiers and paths)
numeric_cols = df_rated.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ['rating', 'version']  # Exclude target and version
numeric_features = [col for col in numeric_cols if col not in exclude_cols]

print(f"\nNumeric features identified: {len(numeric_features)}")
print("Features:", ', '.join(numeric_features[:10]), "...")

# Boolean features
boolean_cols = [col for col in df_rated.columns if df_rated[col].dtype == 'bool']
print(f"\nBoolean features: {len(boolean_cols)}")
print("Features:", ', '.join(boolean_cols[:10]), "...")

print("\n" + "="*80)
print("4. CORRELATION ANALYSIS")
print("="*80)

# Calculate correlations with rating
correlations = df_rated[numeric_features].corrwith(df_rated['rating']).abs().sort_values(ascending=False)
top_correlations = correlations.head(15)

print("\nTop 15 Features Correlated with Rating:")
print("-" * 60)
for i, (feature, corr) in enumerate(top_correlations.items(), 1):
    print(f"{i:2}. {feature:30} | Correlation: {corr:.4f}")

# Visualize top correlations
plt.figure(figsize=(12, 8))
top_correlations.plot(kind='barh', color='steelblue')
plt.xlabel('Absolute Correlation with Rating')
plt.title('Top 15 Features Most Correlated with Song Rating')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('top_correlations.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: top_correlations.png")
plt.close()

print("\n" + "="*80)
print("5. FEATURE IMPORTANCE (Random Forest)")
print("="*80)

# Prepare data for Random Forest (handle missing values)
features_for_rf = top_correlations.head(20).index.tolist()
X = df_rated[features_for_rf].fillna(df_rated[features_for_rf].median())
y = df_rated['rating']

# Train Random Forest
print("\nTraining Random Forest model...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
rf.fit(X, y)

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': features_for_rf,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features (Random Forest):")
print("-" * 60)
for i, row in feature_importance.head(15).iterrows():
    print(f"{row.name+1:2}. {row['feature']:30} | Importance: {row['importance']:.4f}")

# Visualize feature importances
plt.figure(figsize=(12, 8))
feature_importance.head(15).set_index('feature')['importance'].plot(kind='barh', color='coral')
plt.xlabel('Feature Importance')
plt.title('Top 15 Most Important Features (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_rf.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: feature_importance_rf.png")
plt.close()

print("\n" + "="*80)
print("6. BOOLEAN FEATURES ANALYSIS")
print("="*80)

# Analyze boolean features
boolean_impact = {}
for col in boolean_cols[:15]:  # Limit to first 15 for speed
    if col in df_rated.columns:
        true_ratings = df_rated[df_rated[col] == True]['rating']
        false_ratings = df_rated[df_rated[col] == False]['rating']
        
        if len(true_ratings) > 0 and len(false_ratings) > 0:
            mean_diff = true_ratings.mean() - false_ratings.mean()
            # T-test
            t_stat, p_value = stats.ttest_ind(true_ratings, false_ratings, equal_var=False)
            boolean_impact[col] = {
                'mean_diff': mean_diff,
                'p_value': p_value,
                'true_mean': true_ratings.mean(),
                'false_mean': false_ratings.mean()
            }

boolean_df = pd.DataFrame(boolean_impact).T
boolean_df = boolean_df.sort_values('mean_diff', key=abs, ascending=False)

print("\nBoolean Features Impact on Rating (sorted by absolute mean difference):")
print("-" * 80)
print(f"{'Feature':<30} | {'True Mean':<10} | {'False Mean':<10} | {'Difference':<10} | {'P-Value'}")
print("-" * 80)
for feature, row in boolean_df.head(10).iterrows():
    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
    print(f"{feature:<30} | {row['true_mean']:>10.3f} | {row['false_mean']:>10.3f} | {row['mean_diff']:>+10.3f} | {row['p_value']:.4f} {sig}")

print("\n" + "="*80)
print("7. KEY INSIGHTS & RECOMMENDATIONS")
print("="*80)

print("\n📊 SUMMARY OF KEY FINDINGS:")
print("-" * 80)

print("\n1. MOST IMPORTANT NUMERIC FEATURES:")
top_5_corr = top_correlations.head(5)
for i, (feature, corr) in enumerate(top_5_corr.items(), 1):
    print(f"   {i}. {feature} (correlation: {corr:.4f})")

print("\n2. MOST IMPORTANT FEATURES (Random Forest):")
top_5_rf = feature_importance.head(5)
for i, row in top_5_rf.iterrows():
    print(f"   {i+1}. {row['feature']} (importance: {row['importance']:.4f})")

print("\n3. MOST IMPACTFUL BOOLEAN FEATURES:")
for i, (feature, row) in enumerate(boolean_df.head(5).iterrows(), 1):
    direction = "positive" if row['mean_diff'] > 0 else "negative"
    print(f"   {i}. {feature} ({direction} impact: {row['mean_diff']:+.3f})")

print("\n4. RECOMMENDATIONS FOR MODELING:")
print("   • Focus on engagement metrics (n_favorites, n_views, n_ratings)")
print("   • Include song complexity features (n_notes, song_length)")
print("   • Consider user-related features (is_user_pro, is_official)")
print("   • Audio features (pitch_class_entropy, scale_consistency) show correlation")
print("   • Handle missing values carefully - many features have significant missingness")

print("\n5. DATA QUALITY NOTES:")
print(f"   • Only {len(df_rated)/len(df)*100:.1f}% of songs have ratings")
print(f"   • Many optional fields have high missingness:")
missing_pct = (df_rated[numeric_features].isnull().sum() / len(df_rated) * 100).sort_values(ascending=False)
for feature, pct in missing_pct.head(5).items():
    if pct > 0:
        print(f"     - {feature}: {pct:.1f}% missing")

print("\n" + "="*80)
print("8. CREATING CORRELATION HEATMAP")
print("="*80)

# Create correlation heatmap for top features
top_features_for_heatmap = top_correlations.head(12).index.tolist() + ['rating']
corr_matrix = df_rated[top_features_for_heatmap].corr()

plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap: Top Features vs Rating', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: correlation_heatmap.png")
plt.close()

print("\n" + "="*80)
print("9. SCATTER PLOTS: TOP FEATURES VS RATING")
print("="*80)

# Create scatter plots for top 6 features
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

top_6_features = top_correlations.head(6).index.tolist()
for i, feature in enumerate(top_6_features):
    ax = axes[i]
    # Sample data for faster plotting
    sample_df = df_rated[[feature, 'rating']].dropna()
    if len(sample_df) > 5000:
        sample_df = sample_df.sample(5000, random_state=42)
    
    ax.scatter(sample_df[feature], sample_df['rating'], alpha=0.3, s=10)
    ax.set_xlabel(feature)
    ax.set_ylabel('Rating')
    ax.set_title(f'{feature} vs Rating (r={correlations[feature]:.3f})')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(sample_df[feature], sample_df['rating'], 1)
    p = np.poly1d(z)
    ax.plot(sample_df[feature].sort_values(), p(sample_df[feature].sort_values()), 
            "r--", alpha=0.8, linewidth=2)

plt.tight_layout()
plt.savefig('scatter_plots_top_features.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: scatter_plots_top_features.png")
plt.close()

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated Files:")
print("  1. rating_distribution.png - Distribution of song ratings")
print("  2. top_correlations.png - Top features correlated with ratings")  
print("  3. feature_importance_rf.png - Random Forest feature importances")
print("  4. correlation_heatmap.png - Correlation matrix heatmap")
print("  5. scatter_plots_top_features.png - Scatter plots of top features")
print("\n" + "="*80)
