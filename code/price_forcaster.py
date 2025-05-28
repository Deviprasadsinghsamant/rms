import pandas as pd
import numpy as np
import datetime as dt
from demand_features import rf_cols
from demand import  model_demand

def forecast_hotel_prices(hotel_info, df_sim, as_of_date):
    """
    Wrapper function around model_demand that forecasts hotel prices
    using the existing implementation that's proven to work.
    
    Parameters:
    -----------
    hotel_info : dict
        Dictionary containing hotel information:
        - hotel_num: int (1 or 2)
        
    df_sim : pandas DataFrame
        DataFrame containing historical booking data and features
        
    as_of_date : str
        The reference date for predictions in YYYY-MM-DD format
        
    Returns:
    --------
    forecast_df : pandas DataFrame
        DataFrame containing forecasted dates, recommended prices, and metrics
    """
    
    # Extract hotel number
    hotel_num = hotel_info.get('hotel_num')
    assert hotel_num in (1, 2), ValueError("hotel_num must be (int) 1 or 2.")
    
    # Ensure as_of_date is in the right format
    try:
        as_of_date = pd.to_datetime(as_of_date).strftime("%Y-%m-%d")
    except:
        raise ValueError("as_of_date must be in YYYY-MM-DD format")
    
    # Get capacity for this hotel
    # capacity = 187 if hotel_num == 1 else 226
    
    # print(f"Forecasting prices for hotel {hotel_num} as of {as_of_date}")
    # print(f"Input data shape: {df_sim.shape}")
    
    # Call the existing model_demand function that works
    try:
        # This function does the heavy lifting and is known to work
        df_demand = model_demand(hotel_num, df_sim, as_of_date)
        print(f"Demand modeling successful. Result shape: {df_demand.shape}")
    except Exception as e:
        print(f"Error in model_demand: {str(e)}")
        # Try to provide more context about the error
        import traceback
        traceback.print_exc()
        raise
    
    # Create a more readable forecast DataFrame
    try:
        # Select the most important columns
        forecast_columns = [
            'StayDate', 'DOW', 'SellingPrice', 'OptimalRate', 'RecommendedPriceChange',
            'Proj_TRN_RemDemand', 'TRN_rnPU_AtOptimal', 'TRN_RevPU_AtOptimal', 
            'ProjRN_ChgAtOptimal', 'ProjRevChgAtOptimal', 'ProjOcc'
        ]
        
        # Filter to only columns that exist
        available_columns = [col for col in forecast_columns if col in df_demand.columns]
        forecast_df = df_demand[available_columns].copy()
        
        # Rename columns for clarity
        column_mapping = {
            'StayDate': 'Date',
            'DOW': 'DayOfWeek',
            'SellingPrice': 'CurrentPrice',
            'OptimalRate': 'RecommendedPrice',
            'RecommendedPriceChange': 'PriceChange',
            'Proj_TRN_RemDemand': 'ProjectedDemand',
            'TRN_rnPU_AtOptimal': 'RoomNightsAtOptimal',
            'TRN_RevPU_AtOptimal': 'RevenueAtOptimal',
            'ProjRN_ChgAtOptimal': 'RoomNightChange',
            'ProjRevChgAtOptimal': 'RevenueChange',
            'ProjOcc': 'ProjectedOccupancy'
        }
        
        # Only rename columns that exist
        rename_dict = {k: v for k, v in column_mapping.items() if k in forecast_df.columns}
        forecast_df = forecast_df.rename(columns=rename_dict)
        
        # Sort by date for better readability
        if 'Date' in forecast_df.columns:
            forecast_df = forecast_df.sort_values('Date')
        
        print("Forecast summary created successfully.")
    except Exception as e:
        print(f"Error creating forecast summary: {str(e)}")
        # Return the raw results if we can't create the summary
        return df_demand
    
    # Calculate summary metrics for reporting
    try:
        if 'PriceChange' in forecast_df.columns:
            avg_price_change = round(forecast_df['PriceChange'].mean(), 2)
            print(f"\nAverage Recommended Price Change: ${avg_price_change}")
        
        if 'RevenueChange' in forecast_df.columns:
            total_revenue_gain = round(forecast_df['RevenueChange'].sum(), 2)
            print(f"Total Projected Revenue Gain: ${total_revenue_gain}")
        
        if 'ProjectedOccupancy' in forecast_df.columns:
            avg_occupancy = round(forecast_df['ProjectedOccupancy'].mean() * 100, 1)
            print(f"Average Projected Occupancy: {avg_occupancy}%")
    except Exception as e:
        print(f"Warning: Could not calculate summary metrics - {str(e)}")
    
    print("\nPrice forecasting completed successfully!")
    return forecast_df


# Example usage:
"""
# Sample hotel information
hotel_info = {
    'hotel_num': 1  # Hotel number (1 or 2)
}

# Reference date for predictions
as_of_date = "2023-05-01"

# Get the price forecast
forecast_results = forecast_hotel_prices(hotel_info, df_sim, as_of_date)

# Display the first few days of the forecast
print(forecast_results.head())
"""




hotel_info = {
    'hotel_num': 1,  # Hotel number (1 or 2)
    'capacity': 187  # Optional, will use default if not provided
}

# Reference date for predictions
as_of_date = "2016-05-01"
df_sim = pd.read_csv(r"C:\Users\Quotus\Desktop\rms002\rms_init\hotel-revman-system\data\h1_stats.csv")

# Get the price forecast
forecast_results = forecast_hotel_prices(hotel_info, df_sim, as_of_date)

# Display the first few days of the forecast
print(forecast_results.head())
print(forecast_results.columns)
print(forecast_results.shape)
