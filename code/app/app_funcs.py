import pandas as pd
import os
from app_utils import renamed_user_cols

DATE_FMT = "%Y-%m-%d"


def get_pricing(hotel_num, cols):
    try:
        if hotel_num == 1:
            capacity = 187
        else:
            capacity = 226
            
        filepath = f"../../data/results/h{hotel_num}_pricing_v2.csv"
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pricing file not found: {filepath}")
            
        df = pd.read_csv(
            filepath,
            parse_dates=["AsOfDate", "StayDate"],
        )
        df.index = pd.DatetimeIndex(df.StayDate).date
        df = df.sort_index()
        df["LYA_Occ"] = df["LYA_RoomsSold"] / capacity
        df.drop(columns="Unnamed: 0", errors='ignore', inplace=True)
        df = df[cols]
        return df
        
    except Exception as e:
        print(f"Error loading pricing data: {str(e)}")
        return None
