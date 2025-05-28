"""
This script trains a model to predict remaining transient pickup (demand).

It then finds the optimal selling price based on resulting room revenue.

Returns a DataFrame containing 31 days of future dates, along with predicted demand
at the optimal selling prices.
"""
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error # type: ignore

from demand_features import rf_cols

DATE_FMT = "%Y-%m-%d"


def splits(df_sim, features, train_size=0.7, random_state=42):
    """
    Splits df_sim into X_train, X_test, y_train, y_test based on percentage.
    
    Parameters:
    -----------
    df_sim : pandas DataFrame
        The DataFrame containing the data
    features : list
        List of feature column names
    train_size : float, default=0.7
        Proportion of the dataset to include in the train split (0 to 1)
    random_state : int, default=42
        Controls the shuffling applied to the data before applying the split
        
    Returns:
    --------
    X_train, y_train, X_test, y_test : DataFrames/Series for training and testing
    """
    from sklearn.model_selection import train_test_split
    
    # Split the data using sklearn's train_test_split
    df_train, df_test = train_test_split(
        df_sim, 
        train_size=train_size,
        random_state=random_state
    )
    
    # Extract features and target variables
    X_train = df_train[features].copy()
    X_test = df_test[features].copy()
    y_train = df_train["ACTUAL_TRN_RoomsPickup"].copy()
    y_test = df_test["ACTUAL_TRN_RoomsPickup"].copy()
    
    print(f"Training set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    
    return X_train, y_train, X_test, y_test

def train_model(
    df_sim, as_of_date, hotel_num, features, X_train, y_train, X_test, y_test
):
    print("X_train",X_train)
    print("X_test",X_test)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    rfm = RandomForestRegressor( 
        n_estimators=550,
        n_jobs=-1,
        random_state=20,
    )
    rfm.fit(X_train, y_train)
    print("x test",X_test)  
    preds = rfm.predict(X_test)
    print(preds)
    # add preds back to original
    X_test["Proj_TRN_RemDemand"] = preds.round(0).astype(int)

    mask = df_sim["AsOfDate"] == as_of_date
    df_demand = df_sim[mask].copy()
    df_demand["Proj_TRN_RemDemand"] = X_test["Proj_TRN_RemDemand"]

    return df_demand, rfm, preds

def calculate_rev_at_price(price, df_demand, model, df_index, features):
    """
    Calculates transient room revenue at predicted selling prices.
    
    Parameters:
    -----------
    price : float
        The selling price to test
    df_demand : pandas DataFrame
        DataFrame containing demand data
    model : trained model
        The predictive model to use
    df_index : int
        Index of the row to modify
    features : list
        List of feature column names
        
    Returns:
    --------
    resulting_rn : float
        Predicted room nights
    resulting_rev : float
        Predicted revenue (rn * price)
    """
    # Create a copy to avoid modifying the original
    df = df_demand.copy()
    
    # Update the selling price at the specified index
    df.loc[df_index, "SellingPrice"] = price
    
    # Extract feature values directly as a list and ensure they're all numeric
    feature_values = []
    for col in features:
        val = df.loc[df_index, col]
        # Convert to float - handles any non-numeric values
        try:
            val = float(val)
        except (ValueError, TypeError):
            val = 0.0  # Default for non-convertible values
        feature_values.append(val)
    
    # Convert to DataFrame to preserve feature names
    X = pd.DataFrame([feature_values], columns=features)
    
    # Make prediction and calculate revenue 
    resulting_rn = model.predict(X)[0]
    resulting_rev = round(resulting_rn * price, 2)
    
    return resulting_rn, resulting_rev

def get_optimal_prices(df_demand, as_of_date, model, features):
    """
    Models demand at current prices & stores resulting TRN RoomsBooked & Rev.
    Then adjusts prices by 5% increments in both directions, up to 25%.
    
    Parameters:
    -----------
    df_demand : pandas DataFrame
        DataFrame containing demand data
    as_of_date : datetime
        The reference date for predictions
    model : trained model
        The predictive model to use
    features : list
        List of feature column names
        
    Returns:
    --------
    df_demand : pandas DataFrame
        Updated dataframe
    optimal_prices : list
        List of optimal price tuples
    """
    # Get indices of rows to process
    indices = list(df_demand.index)
    # Define price adjustment percentages (excluding 0%)
    price_adjustments = np.delete(
        np.arange(-0.25, 0.30, 0.05).round(2), 5
    )  # delete zero (already have it)
    
    optimal_prices = []
    print("df_demand",df_demand.head())
    # Process each date
    for i in indices:
        try:
            # Ensure all features are available for this row
            feature_data = df_demand.loc[i, features].copy()
            print("feature_data",feature_data)
            # First ensure all feature values are numeric for this row
            for col in features:
                if col in df_demand.columns:
                    # Try to convert to numeric, coerce errors to NaN
                    if not pd.api.types.is_numeric_dtype(df_demand[col].dtype):
                        df_demand.loc[i, col] = pd.to_numeric(df_demand.loc[i, col], errors='coerce')
            
            # Now check for and fill NaN values
            has_nan_features = pd.isna(feature_data).any()
            if has_nan_features:
                # print(f"Filling NaN values in features for index {i}")
                for col in features:
                    if pd.isna(df_demand.loc[i, col]):
                        if pd.api.types.is_numeric_dtype(df_demand[col].dtype):
                            df_demand.loc[i, col] = df_demand[col].mean() if not pd.isna(df_demand[col].mean()) else 0
                        else:
                            df_demand.loc[i, col] = 0
            
            # Calculate baseline metrics at original price
            original_rate = round(df_demand.loc[i, "SellingPrice"], 2)
            # print("original rate",original_rate)
            # Prepare features for prediction and ensure all are numeric
            feature_values = []
            for col in features:
                val = df_demand.loc[i, col]
                # Convert to float - handles any remaining non-numeric values
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    val = 0.0  # Default for non-convertible values
                feature_values.append(val)
                
            # Convert to DataFrame to preserve feature names
            date_X = pd.DataFrame([feature_values], columns=features)
            
            # Predict demand at original price
            original_rn = model.predict(date_X)[0]
            original_rev = original_rn * original_rate
            
            # Initialize optimal values with original values
            optimal_rate = (
                original_rate,  # will be updated - optimal rate
                original_rn,    # will be updated - predicted RN at optimal
                original_rev,   # will be updated - predicted rev at optimal
                original_rate,  # will remain - original rate
                original_rn,    # will remain - predicted RN at original
                original_rev,   # will remain - predicted rev at original
            )

            # Try different price points to maximize revenue
            for pct in price_adjustments:
                new_rate = round(original_rate * (1 + pct), 2)
                resulting_rn, resulting_rev = calculate_rev_at_price(
                    new_rate, df_demand, model, i, features
                )

                # Update optimal if this price produces higher revenue
                if resulting_rev > optimal_rate[2]:  # Compare with optimal_rev (index 2)
                    optimal_rate = (
                        new_rate,
                        resulting_rn,
                        resulting_rev,
                        original_rate,
                        original_rn,
                        original_rev,
                    )
        
        except Exception as e:
            print(f"Error processing index {i}: {str(e)}")
            # Use original values as fallback
            original_rate = round(df_demand.loc[i, "SellingPrice"], 2)
            optimal_rate = (
                original_rate,
                0,  # Placeholder for RN
                0,  # Placeholder for revenue
                original_rate,
                0,
                0,
            )
            
        optimal_prices.append(optimal_rate)
    
    # Verify we have the expected number of results
    expected_count = len(indices)
    actual_count = len(optimal_prices)
    if actual_count != expected_count:
        print(f"Warning: Expected {expected_count} price points, but got {actual_count}")
    print("df demand",df_demand)
    print("optiaml price",optimal_prices)
    return df_demand, optimal_prices

def add_rates(df_demand, optimal_prices):
    """
    Implements price recommendations from optimize_price and returns pricing_df
    """
    new_rates = []
    resulting_rns = []
    resulting_revs = []
    original_rates = []
    original_rns = []
    original_revs = []
    for (
        new_rate,
        resulting_rn,
        resulting_rev,
        original_rate,
        original_rn,
        original_rev,
    ) in optimal_prices:
        new_rates.append(new_rate)
        resulting_rns.append(resulting_rn)
        resulting_revs.append(round(resulting_rev, 2))
        original_rates.append(original_rate)
        original_rns.append(original_rn)
        original_revs.append(round(original_rev, 2))

    df_demand["OptimalRate"] = new_rates
    df_demand["TRN_rnPU_AtOptimal"] = resulting_rns
    df_demand["TRN_RevPU_AtOptimal"] = resulting_revs
    df_demand["TRN_rnPU_AtOriginal"] = original_rns
    df_demand["TRN_RN_ProjVsActual_OP"] = (
        df_demand["TRN_rnPU_AtOriginal"] - df_demand["ACTUAL_TRN_RoomsPickup"]
    )
    df_demand["TRN_RevPU_AtOriginal"] = original_revs
    df_demand["TRN_RevProjVsActual_OP"] = (
        df_demand["TRN_RevPU_AtOriginal"] - df_demand["ACTUAL_TRN_RevPickup"]
    )

    return df_demand

def summarize_model_results(model, y_test, preds):
    """Writes model metrics to STDOUT."""
    r2 = round(r2_score(y_test, preds), 3)
    mae = round(mean_absolute_error(y_test, preds), 3)
    mse = round(mean_squared_error(y_test, preds), 3)

    print(
        f"R² score on test set (stay dates Aug 1 - Aug 31, 2017):                        {r2}"
    )
    print(
        f"MAE (Mean Absolute Error) score on test set (stay dates Aug 1 - Aug 31, 2017): {mae}"
    )
    print(
        f"MSE (Mean Squared Error) score on test set (stay dates Aug 1 - Aug 31, 2017):  {mse}\n"
    )
    pass

def debug_datetime_column(df, column_name="StayDate"):
    """
    Helper function to diagnose issues with datetime columns
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the datetime column to diagnose
    column_name : str, default="StayDate"
        Name of the datetime column to check
        
    Returns:
    --------
    None - prints diagnostic information
    """
    
    if column_name not in df.columns:
        print(f"ERROR: Column '{column_name}' not found in DataFrame")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Basic info
    print(f"\n--- Diagnostic info for '{column_name}' column ---")
    print(f"Data type: {df[column_name].dtype}")
    print(f"Sample values: {df[column_name].head().tolist()}")
    
    # Missing value analysis
    missing = df[column_name].isna().sum()
    print(f"Missing values: {missing} ({missing/len(df)*100:.2f}%)")
    
    # Type analysis
    if missing < len(df):
        non_null_values = df[column_name].dropna()
        value_types = [type(x).__name__ for x in non_null_values]
        type_counts = pd.Series(value_types).value_counts()
        print("Value types in column:")
        print(type_counts)
    
    # Fix attempt
    print("\n--- Attempting to fix ---")
    try:
        # Try to convert to datetime
        fixed_series = pd.to_datetime(df[column_name], errors='coerce')
        new_missing = fixed_series.isna().sum()
        newly_missing = new_missing - missing
        
        print(f"After conversion: {new_missing} missing values ({newly_missing} new NaT values)")
        
        if newly_missing > 0:
            # Show problematic values
            problem_indices = df.index[fixed_series.isna() & df[column_name].notna()]
            print("Sample problematic values:")
            for idx in problem_indices[:5]:  # Show first 5 problematic values
                print(f"  Index {idx}: '{df.loc[idx, column_name]}'")
        
        print("\nRECOMMENDATION:")
        if newly_missing == 0:
            print("✅ Safe to convert using: df['StayDate'] = pd.to_datetime(df['StayDate'])")
        elif newly_missing / len(df) < 0.05:
            print("⚠️ Minor issues - convert with: df['StayDate'] = pd.to_datetime(df['StayDate'], errors='coerce')")
        else:
            print("❌ Major conversion issues - manual inspection needed")
            
    except Exception as e:
        print(f"Error during conversion attempt: {str(e)}")
        print("❌ Custom parsing may be required")
    
    return

def add_display_columns(df_demand, capacity):
    """
    Adds informative columns that will be displayed to app users:
        - RecommendedPriceChange (optimal rate variance to original rate)
        - ProjChgAtOptimal (projected RN & Rev change at optimal rates)
        - DOW (day of week)
        - Actual & Projected Occ
        
    Parameters:
    -----------
    df_demand : pandas DataFrame
        DataFrame containing demand data
    capacity : int
        Hotel capacity (total number of rooms)
        
    Returns:
    --------
    df_demand : pandas DataFrame
        DataFrame with additional display columns
    """
    import datetime as dt
    import pandas as pd
    
    # Calculate price change recommendation
    df_demand["RecommendedPriceChange"] = (
        df_demand["OptimalRate"] - df_demand["SellingPrice"]
    )

    # Calculate projected changes in room nights and revenue
    df_demand["ProjRN_ChgAtOptimal"] = (
        df_demand["TRN_rnPU_AtOptimal"] - df_demand["TRN_rnPU_AtOriginal"]
    )

    df_demand["ProjRevChgAtOptimal"] = (
        df_demand["TRN_RevPU_AtOptimal"] - df_demand["TRN_RevPU_AtOriginal"]
    )

    # Add day of week (DOW) column safely
    try:    
        # First ensure StayDate is datetime type
        if df_demand["StayDate"].dtype != 'datetime64[ns]':
            print("Converting StayDate to datetime format...")
            df_demand["StayDate"] = pd.to_datetime(df_demand["StayDate"], errors='coerce')
        
        # Handle any remaining NaT values
        if df_demand["StayDate"].isna().any():
            print(f"Warning: {df_demand['StayDate'].isna().sum()} NaT values found in StayDate column")
            # For missing dates, use a placeholder value like "N/A"
            df_demand["DOW"] = df_demand["StayDate"].apply(
                lambda x: dt.datetime.strftime(x, format="%a") if pd.notnull(x) else "N/A"
            ).astype(str)
        else:
            # If no NaN values, proceed normally
            df_demand["DOW"] = df_demand["StayDate"].apply(
                lambda x: dt.datetime.strftime(x, format="%a")
            ).astype(str)
            
    except Exception as e:
        print(f"Error processing StayDate column: {str(e)}")
        print("Using fallback method for DOW calculation...")
        # Fallback method: just add empty DOW column
        df_demand["DOW"] = "N/A"

    # Calculate and print summary metrics
    try:
        avg_price_change = round(df_demand["RecommendedPriceChange"].mean(), 2)
        total_rn_opp = round(df_demand["ProjRN_ChgAtOptimal"].sum(), 2)
        total_rev_opp = round(df_demand["ProjRevChgAtOptimal"].sum(), 2)

        print(f"Average recommended price change: {avg_price_change}")
        print(f"Estimated RN (Roomnight) growth: {total_rn_opp}")
        print(f"Estimated revenue growth: {total_rev_opp}")
    except Exception as e:
        print(f"Warning: Could not calculate summary metrics - {str(e)}")

    # Calculate occupancy columns
    try:
        # Actual occupancy
        df_demand["ACTUAL_Occ"] = round(df_demand["ACTUAL_RoomsSold"] / capacity, 2)
        
        # Projected total rooms sold
        df_demand["TotalProjRoomsSold"] = (
            capacity - df_demand["RemSupply"] + df_demand["Proj_TRN_RemDemand"]
        )
        
        # Projected occupancy
        df_demand["ProjOcc"] = round(df_demand["TotalProjRoomsSold"] / capacity, 2)
    except Exception as e:
        print(f"Warning: Could not calculate occupancy metrics - {str(e)}")
        # Set default values if calculation fails
        if "ACTUAL_Occ" not in df_demand.columns:
            df_demand["ACTUAL_Occ"] = 0
        if "TotalProjRoomsSold" not in df_demand.columns:
            df_demand["TotalProjRoomsSold"] = 0
        if "ProjOcc" not in df_demand.columns:
            df_demand["ProjOcc"] = 0

    return df_demand.copy()

def model_demand(hotel_num, df_sim, as_of_date):
    """
    This function models demand for each date in df_sim. df_sim is the output of agg.py.
    """
    assert hotel_num in (1, 2), ValueError("hotel_num must be (int) 1 or 2.")
    if hotel_num == 1:
        capacity = 187
    else:
        capacity = 226

    # First, ensure all data types are compatible
    print("Converting features to appropriate types...")
    for col in rf_cols:
        if col in df_sim.columns:
            # Convert boolean columns to int (0/1)
            if df_sim[col].dtype == bool:
                df_sim[col] = df_sim[col].astype(int)
            # Try to convert other columns to float
            elif df_sim[col].dtype not in ['float64', 'int64']:
                df_sim[col] = pd.to_numeric(df_sim[col], errors='coerce')
            # Fill any NaN values with 0
            if df_sim[col].isna().any():
                print(f"Column {col} has {df_sim[col].isna().sum()} NaN values")
                df_sim[col] = df_sim[col].fillna(0)
    
    # Also ensure the target variable is properly formatted
    if "ACTUAL_TRN_RoomsPickup" in df_sim.columns and df_sim["ACTUAL_TRN_RoomsPickup"].dtype not in ['float64', 'int64']:
        df_sim["ACTUAL_TRN_RoomsPickup"] = pd.to_numeric(df_sim["ACTUAL_TRN_RoomsPickup"], errors='coerce').fillna(0)

    print("Training Random Forest model to predict remaining transient demand...")
    X_train, y_train, X_test, y_test = splits(df_sim, rf_cols)
    # print("Rf cols",rf_cols)
    print("Splitting data into training and test sets...")
    df_demand, model, preds = train_model(
        df_sim, as_of_date, hotel_num, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, features=rf_cols
    )

    print("Model ready.\n")
    summarize_model_results(model, y_test, preds)

    # print("Calculating optimal selling prices...\n")
    # df_demand, optimal_prices = get_optimal_prices(
    #     df_demand, as_of_date, model, rf_cols
    # )
    # df_demand = add_rates(df_demand, optimal_prices)
    # df_demand = add_display_columns(df_demand, capacity)

    print("Simulation ready.\n")

    return df_demand