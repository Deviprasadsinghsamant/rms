# Hotel Revenue Management System Code Analysis

## Price Prediction Analysis

After analyzing the codebase, there is no direct room price prediction model. Instead, the system:

1. Uses a RandomForest model to predict demand
2. Tests different price points (-25% to +25% of current price) 
3. Picks the price that maximizes predicted revenue

Key files involved in pricing:

### demand.py
- Main pricing logic in `get_optimal_prices()` function
- Tests price adjustments from -25% to +25% in 5% increments 
- Calculates revenue at each price point using predicted demand
- Does not predict prices directly, only optimizes based on demand predictions

## File Purposes

### Python Files

1. **demand.py**
- Predicts transient room demand using RandomForest
- Optimizes room rates based on predicted demand
- Core file for revenue management logic

2. **model_tools.py**  
- Helper functions for model evaluation
- Contains confusion matrix visualization
- Statistical metric calculations

3. **model_cancellations.py**
- Predicts reservation cancellations
- Uses XGBoost classifier
- Helps adjust inventory based on predicted cancellations

4. **demand_features.py**
- Defines feature lists for demand modeling
- Contains column definitions for Hotel 1 and Hotel 2

### Jupyter Notebooks

1. **demand_model_selection.ipynb**
- Model selection process for Hotel 1
- Tests different algorithms (Linear Regression, Random Forest, XGBoost)
- Documents hyperparameter tuning

2. **demand_model_selection_h2.ipynb**
- Same as above but for Hotel 2
- Shows different model parameters needed for city hotel

3. **demand_model_analysis_h1.ipynb & demand_model_analysis_h2.ipynb**
- Analyze model performance
- Visualize predictions vs actuals
- Error analysis by day of week

4. **_reproduce.ipynb**
- Main notebook to reproduce the analysis
- Step-by-step instructions
- Data preparation and model execution

5. **agg_test.ipynb**
- Tests data aggregation functions
- Validates feature engineering
- Plots historical data

6. **snippets.ipynb**
- Collection of code snippets
- Helper functions
- Not meant to be run directly

## Summary
The system does not directly predict room prices. Instead, it:
1. Predicts future demand
2. Tests multiple price points 
3. Selects price that maximizes revenue based on predicted demand

The approach is more akin to price optimization rather than price prediction.