"""This script aggregates the data in ../data/otb-data/ folder
that was generated from save_sims.py.

It then adds in calculated statistics/features that will be used
for modeling demand.

Note: must be at least 364 days of actuals for training; the rest
future-looking starting with t+0 on AOD (as-of date).
"""

import datetime as dt
import pandas as pd
import numpy as np
import os

from agg_utils import stly_cols_agg, ly_cols_agg, trash_can, pace_tuples, gap_tuples

DATE_FMT = "%Y-%m-%d"
SIM_AOD = dt.date(2017, 8, 1)  # simulation as-of date
# FOLDER = r"C:\Users\Quotus\Desktop\rms002\rms_init\hotel-revman-system\code\sims2\\"
H1_CAPACITY = 187
H2_CAPACITY = 226
# H1_DBD = pd.read_pickle(r"C:\Users\Quotus\Desktop\rms002\rms_init\hotel-revman-system\code\pickle\h1_dbd.pick")
# H2_DBD = p



# --- Dynamic Path Resolution ---
# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Construct the path to the 'sims2' output folder
# This assumes 'sims2' is a sibling of the 'code' directory,
# or more precisely, it's relative to where the script is run from.
# If 'sims2' is inside 'code', adjust to: os.path.join(script_dir, "sims2")
# Based on your previous context, it might be better to assume it's like '../sims2' from 'code'.
# Let's assume 'sims2' is at: ~/Desktop/rms/rms/sims2/
# And your script is at: ~/Desktop/rms/rms/code/your_script.py
# So, from 'code' you need to go up one level and then down into 'sims2'.
base_dir = os.path.dirname(script_dir) # Go up one level from 'code' (to 'rms' folder)
FOLDER = os.path.join(base_dir, "sims2", "") # Join with 'sims2' and ensure trailing slash

# Construct the path to the 'pickle' directory
pickle_dir = os.path.join(script_dir, "pickle") # Assuming 'pickle' is inside 'code'

# --- Load Data ---
H1_DBD = pd.read_pickle(os.path.join(pickle_dir, "h1_dbd.pick"))
H2_DBD = pd.read_pickle(os.path.join(pickle_dir, "h2_dbd.pick"))

def combine_files(hotel_num, sim_aod, prelim_csv_out=None):
    """Combines all required files in FOLDER into one DataFrame."""
    sim_start = sim_aod - pd.DateOffset(365 * 2)
    lam_include = (
    lambda x: x[:9] == "h" + str(hotel_num) + "_sim_20"
    and pd.to_datetime(x[7:17]) >= sim_start
    and x[7] == "2"
    )
    otb_files = [f for f in os.listdir(FOLDER) if lam_include(f)]
    otb_files.sort()
    
    dfs = []
    for otb_data in otb_files:
        dfs.append(pd.read_pickle(FOLDER + otb_data))
    
    if dfs:
        df_sim = pd.concat(dfs, ignore_index=True)
    else:
        print(f"Warning: No files found for hotel {hotel_num} in {FOLDER}. Returning empty DataFrame.")
        df_sim = pd.DataFrame()

    if prelim_csv_out is not None:
        df_sim.to_csv(prelim_csv_out)
        print(f"'{prelim_csv_out}' file saved.")

    # Print column names to help with debugging
    # print("Columns in df_sim after loading:", df_sim.columns.tolist())
    
    # Check if required columns exist
    required_columns = ['Date', 'STLY_Date', 'DaysUntilArrival']
    missing_columns = [col for col in required_columns if col not in df_sim.columns]
    
    if missing_columns:
        print(f"WARNING: Missing required columns: {missing_columns}")
        # Create DaysUntilArrival if it doesn't exist (assuming default value or calculation)
        if 'DaysUntilArrival' in missing_columns:
            print("Adding placeholder 'DaysUntilArrival' column with default value of 0")
            df_sim['DaysUntilArrival'] = 0  # Replace with appropriate calculation if available

    return df_sim.copy()

def extract_features(df_sim, df_dbd, capacity):
    """A series of functions that add TY (This Year) features to df_sim."""

    def add_aod(df_sim):
        """Adds "AsOfDate" and "STLY_AsOfDate" columns."""
        # print("Columns in df_sim at start of add_aod:", df_sim.columns.tolist())
        
        # Check if required columns exist
        if 'DaysUntilArrival' not in df_sim.columns:
            print("WARNING: 'DaysUntilArrival' column missing. Adding with default value 0.")
            df_sim['DaysUntilArrival'] = 0  # Set default value
        
        if 'Date' not in df_sim.columns or 'STLY_Date' not in df_sim.columns:
            print("ERROR: Required date columns missing. Cannot continue.")
            return df_sim
        
        def apply_aod(row):
            stay_date = pd.to_datetime(row["Date"])
            stly_stay_date = pd.to_datetime(row["STLY_Date"])
            n_days_b4 = int(row["DaysUntilArrival"])
            as_of_date = stay_date - pd.DateOffset(n_days_b4)
            stly_as_of_date = stly_stay_date - pd.DateOffset(n_days_b4)
            return as_of_date, stly_as_of_date

        df_sim[["AsOfDate", "STLY_AsOfDate"]] = df_sim.apply(
            apply_aod, axis=1, result_type="expand"
        )

        df_sim.rename(
            columns={"Date": "StayDate", "STLY_Date": "STLY_StayDate"}, inplace=True
        )

        return df_sim

    def onehot(df_sim):
        ohe_dow = pd.get_dummies(df_sim["DayOfWeek"], drop_first=True)
        dow_ohe_cols = list(ohe_dow.columns)
        df_sim[dow_ohe_cols] = ohe_dow

        return df_sim

    def add_date_info(df_sim):
        # Check if StayDate is already datetime
        if not pd.api.types.is_datetime64_dtype(df_sim["StayDate"]):
            df_sim["StayDate"] = pd.to_datetime(df_sim["StayDate"])
            
        df_sim["MonthNum"] = df_sim.StayDate.dt.month
        df_sim["DayOfWeek"] = df_sim.StayDate.map(
            lambda x: dt.datetime.strftime(x, format="%a")
        )
        # add one-hot-encoded date columns
        df_sim = onehot(df_sim)
        
        # Check if all required columns exist for is_weekend calculation
        required_cols = ["Mon", "Wed", "Tue", "Sun", "Thu"]
        if all(col in df_sim.columns for col in required_cols):
            is_weekend = (
                (df_sim.Mon == 0)
                & (df_sim.Wed == 0)
                & (df_sim.Tue == 0)
                & (df_sim.Sun == 0)
                & (df_sim.Thu == 0)
            )
            df_sim["WE"] = is_weekend
            
            days = ["Mon", "Tue", "Wed", "Thu", "Sat", "Sun"]
            for d in days:
                if d in df_sim.columns:
                    df_sim[d] = df_sim[d].astype("bool")
        else:
            print("WARNING: Not all day columns exist for weekend calculation")
            df_sim["WE"] = False  # Default value
        
        df_sim["week_of_year"] = df_sim.StayDate.dt.isocalendar().week.astype(float)
        return df_sim

    def add_rem_supply(df_sim):
        if 'RoomsOTB' not in df_sim.columns or 'CxlForecast' not in df_sim.columns:
            print("WARNING: Missing columns for RemSupply calculation")
            df_sim["RemSupply"] = capacity  # Default to full capacity
            return df_sim
            
        df_sim["RemSupply"] = (
            capacity - df_sim.RoomsOTB.astype(int) + df_sim.CxlForecast.astype(int)
        )
        return df_sim

    def add_lya(df_sim):
        # Check if required columns exist
        missing_cols = []
        for col in ['RevOTB', 'RoomsOTB', 'TRN_RevOTB', 'TRN_RoomsOTB', 'TRNP_RevOTB', 'TRNP_RoomsOTB']:
            if col not in df_sim.columns:
                missing_cols.append(col)
                
        if missing_cols:
            print(f"WARNING: Missing columns for ADR calculations: {missing_cols}")
            return df_sim
            
        def apply_ly_cols(row):
            try:
                stly_date = row["STLY_StayDate"]
                if pd.isna(stly_date):
                    return tuple(np.zeros(len(ly_cols_agg)))
                    
                # Convert to pandas Timestamp for comparison
                stly_date = pd.Timestamp(stly_date)
                min_date = pd.Timestamp('2015-08-01')
                
                if stly_date < min_date:
                    return tuple(np.zeros(len(ly_cols_agg)))
                    
                stly_date_str = stly_date.strftime(DATE_FMT)
                
                # Handle case where stly_date_str is not in df_dbd index
                if stly_date_str not in df_dbd.index:
                    return tuple(np.zeros(len(ly_cols_agg)))
                    
                df_lya = list(df_dbd.loc[stly_date_str, ly_cols_agg])
                return tuple(df_lya)
            except Exception as e:
                print(f"Error in apply_ly_cols: {e}")
                return tuple(np.zeros(len(ly_cols_agg)))

        # first need ADR OTB
        # Handle division by zero
        df_sim["ADR_OTB"] = df_sim.apply(
            lambda row: round(row["RevOTB"] / row["RoomsOTB"], 2) if row["RoomsOTB"] != 0 else 0, 
            axis=1
        )
        
        df_sim["TRN_ADR_OTB"] = df_sim.apply(
            lambda row: round(row["TRN_RevOTB"] / row["TRN_RoomsOTB"], 2) if row["TRN_RoomsOTB"] != 0 else 0,
            axis=1
        )
        
        df_sim["TRNP_ADR_OTB"] = df_sim.apply(
            lambda row: round(row["TRNP_RevOTB"] / row["TRNP_RoomsOTB"], 2) if row["TRNP_RoomsOTB"] != 0 else 0,
            axis=1
        )

        ly_new_cols = ["LYA_" + col for col in ly_cols_agg]
        
        if 'STLY_StayDate' in df_sim.columns:
            df_sim[ly_new_cols] = df_sim[["STLY_StayDate"]].apply(
                apply_ly_cols, axis=1, result_type="expand"
            )
        else:
            print("WARNING: STLY_StayDate column missing. Cannot calculate LYA values.")
            for col in ly_new_cols:
                df_sim[col] = 0
                
        df_sim.fillna(0, inplace=True)
        return df_sim

    def add_actuals(df_sim):
        actual_cols = [
            "RoomsSold",
            "ADR",
            "RoomRev",
            "TRN_RoomsSold",
            "TRN_ADR",
            "TRN_RoomRev",
            "TRNP_RoomsSold",
            "TRNP_ADR",
            "TRNP_RoomRev",
            "NumCancels",
        ]

        def apply_ty_actuals(row):
            try:
                date = row["StayDate"]
                date_str = dt.datetime.strftime(date, format=DATE_FMT)
                
                if date_str not in df_dbd.index:
                    return tuple(np.zeros(len(actual_cols)))
                    
                results = list(df_dbd.loc[date_str, actual_cols])
                return tuple(results)
            except Exception as e:
                print(f"Error in apply_ty_actuals: {e}")
                return tuple(np.zeros(len(actual_cols)))

        new_actual_cols = ["ACTUAL_" + col for col in actual_cols]
        
        if 'StayDate' in df_sim.columns:
            df_sim[new_actual_cols] = df_sim[["StayDate"]].apply(
                apply_ty_actuals, axis=1, result_type="expand"
            )
            
            # Check required columns before calculations
            required_cols = [
                "ACTUAL_RoomsSold", "RoomsOTB", "ACTUAL_ADR", "ADR_OTB", 
                "ACTUAL_RoomRev", "RevOTB", "ACTUAL_TRN_RoomsSold", "TRN_RoomsOTB",
                "ACTUAL_TRN_ADR", "TRN_ADR_OTB", "ACTUAL_TRN_RoomRev", "TRN_RevOTB",
                "ACTUAL_TRNP_RoomsSold", "TRNP_RoomsOTB", "ACTUAL_TRNP_ADR", 
                "TRNP_ADR_OTB", "ACTUAL_TRNP_RoomRev", "TRNP_RevOTB"
            ]
            
            missing_cols = [col for col in required_cols if col not in df_sim.columns]
            if not missing_cols:
                # add actual pickup
                df_sim["ACTUAL_RoomsPickup"] = df_sim["ACTUAL_RoomsSold"] - df_sim["RoomsOTB"]
                df_sim["ACTUAL_ADR_Pickup"] = df_sim["ACTUAL_ADR"] - df_sim["ADR_OTB"]
                df_sim["ACTUAL_RevPickup"] = df_sim["ACTUAL_RoomRev"] - df_sim["RevOTB"]

                df_sim["ACTUAL_TRN_RoomsPickup"] = (
                    df_sim["ACTUAL_TRN_RoomsSold"] - df_sim["TRN_RoomsOTB"]
                )
                df_sim["ACTUAL_TRN_ADR_Pickup"] = (
                    df_sim["ACTUAL_TRN_ADR"] - df_sim["TRN_ADR_OTB"]
                )
                df_sim["ACTUAL_TRN_RevPickup"] = round(
                    df_sim["ACTUAL_TRN_RoomRev"] - df_sim["TRN_RevOTB"], 2
                )

                df_sim["ACTUAL_TRNP_RoomsPickup"] = (
                    df_sim["ACTUAL_TRNP_RoomsSold"] - df_sim["TRNP_RoomsOTB"]
                )
                df_sim["ACTUAL_TRNP_ADR_Pickup"] = (
                    df_sim["ACTUAL_TRNP_ADR"] - df_sim["TRNP_ADR_OTB"]
                )
                df_sim["ACTUAL_TRNP_RevPickup"] = round(
                    df_sim["ACTUAL_TRNP_RoomRev"] - df_sim["TRNP_RevOTB"], 2
                )
            else:
                print(f"WARNING: Missing columns for ACTUAL calculations: {missing_cols}")
            
        else:
            print("WARNING: StayDate column missing. Cannot calculate ACTUAL values.")
            for col in new_actual_cols:
                df_sim[col] = 0

        df_sim.fillna(0, inplace=True)
        return df_sim

    def add_tminus(df_sim):
        """
        Adds tminus 5, 15, 30 day pickup statistics. Will pull STLY later on to compare
        recent booking vs last year.
        """

        # loop thru tminus windows (for total hotel & TRN) & count bookings
        tms = ["TM30_", "TM15_", "TM05_"]
        segs = ["", "TRN_", "TRNP_"]  # "" for total hotel

        for tm in tms:
            for seg in segs:
                # Check if required columns exist
                tm_rooms_col = tm + seg + "RoomsOTB"
                tm_rev_col = tm + seg + "RevOTB"
                
                if tm_rooms_col not in df_sim.columns or tm_rev_col not in df_sim.columns:
                    print(f"WARNING: Missing {tm}{seg} columns. Skipping calculation.")
                    continue
                
                # add tm_seg_adr - handle division by zero
                df_sim[tm + seg + "ADR_OTB"] = df_sim.apply(
                    lambda row: round(row[tm_rev_col] / row[tm_rooms_col], 2) 
                    if row[tm_rooms_col] != 0 else 0,
                    axis=1
                )
                
                # Check if columns for pickup calculation exist
                if seg + "RoomsOTB" in df_sim.columns and seg + "RevOTB" in df_sim.columns and seg + "ADR_OTB" in df_sim.columns:
                    # and now segmented stats
                    df_sim[tm + seg + "RoomsPickup"] = round(
                        df_sim[seg + "RoomsOTB"] - df_sim[tm + seg + "RoomsOTB"], 2
                    )
                    df_sim[tm + seg + "RevPickup"] = round(
                        df_sim[seg + "RevOTB"] - df_sim[tm + seg + "RevOTB"], 2
                    )
                    df_sim[tm + seg + "ADR_Pickup"] = round(
                        df_sim[seg + "ADR_OTB"] - df_sim[tm + seg + "ADR_OTB"], 2
                    )
                else:
                    print(f"WARNING: Missing columns for {tm}{seg} pickup calculation.")
                    
        return df_sim

    def add_gaps(df_sim):
        # add gap to lya cols
        missing_pairs = []
        for lya, ty_otb, new_col in gap_tuples:
            if lya in df_sim.columns and ty_otb in df_sim.columns:
                df_sim[new_col] = df_sim[lya] - df_sim[ty_otb]
            else:
                missing_pairs.append((lya, ty_otb))
                df_sim[new_col] = 0  # Default value
                
        if missing_pairs:
            print(f"WARNING: Missing column pairs for gap calculation: {missing_pairs}")
            
        return df_sim

    # Fix: Remove commas - these create tuples instead of assignments
    df_sim = add_aod(df_sim)
    df_sim = add_date_info(df_sim)
    df_sim = add_rem_supply(df_sim)
    df_sim = add_lya(df_sim)
    df_sim = add_actuals(df_sim)
    df_sim = add_tminus(df_sim)
    df_sim = add_gaps(df_sim)
    
    return df_sim


def merge_stly(df_sim):
    """
    For each as_of_date + stay_date combo in df_sim from 2016-08-02 to 2017-08-31,
    pull the corresponding 2015-2016 (same-time last year, adjusted for DOW).
    """

    def add_stly_otb(df_sim):
        # Check if required columns exist
        required_cols = ['AsOfDate', 'StayDate', 'STLY_AsOfDate', 'STLY_StayDate']
        missing_cols = [col for col in required_cols if col not in df_sim.columns]
        
        if missing_cols:
            print(f"WARNING: Missing columns for stly_otb calculation: {missing_cols}")
            return df_sim
            
        # first, create unique ID col (id) for each as-of-date/stay-date combo
        # then, add a stly_id column that we can use as left key for our merge (self-join)
        df_sim_ids = df_sim.AsOfDate.astype(str) + " - " + df_sim.StayDate.astype(str)
        df_sim_stly_ids = (
            df_sim.STLY_AsOfDate.astype(str) + " - " + df_sim.STLY_StayDate.astype(str)
        )
        df_sim.insert(0, "id", df_sim_ids)
        df_sim.insert(1, "stly_id", df_sim_stly_ids)

        # Verify all required columns for merge exist
        if not all(col in df_sim.columns for col in stly_cols_agg):
            print(f"WARNING: Not all stly_cols_agg columns exist in df_sim")
            missing = [col for col in stly_cols_agg if col not in df_sim.columns]
            print(f"Missing columns: {missing}")
            
            # Create missing columns with default values
            for col in missing:
                df_sim[col] = 0
                
        # Perform merge
        try:
            df_sim = df_sim.merge(
                df_sim[stly_cols_agg],
                left_on="stly_id",
                right_on="id",
                suffixes=(None, "_STLY"),
                how="left"  # Use left join to keep all rows
            )
            # Fill NaN values from the merge
            df_sim.fillna(0, inplace=True)
        except Exception as e:
            print(f"Error in merge operation: {e}")
            
        return df_sim

    def add_pace(df_sim):
        # pace_tuples example: ('RoomsOTB', 'RoomsOTB_STLY' )
        missing_pairs = []
        for ty_stat, stly_stat in pace_tuples:
            if ty_stat in df_sim.columns and stly_stat in df_sim.columns:
                new_stat_name = ty_stat
                if ty_stat.startswith("ACTUAL_"):
                    new_stat_name = ty_stat[8:]
                df_sim["Pace_" + new_stat_name] = df_sim[ty_stat] - df_sim[stly_stat]
            else:
                missing_pairs.append((ty_stat, stly_stat))
                df_sim["Pace_" + (ty_stat[8:] if ty_stat.startswith("ACTUAL_") else ty_stat)] = 0
                
        if missing_pairs:
            print(f"WARNING: Missing column pairs for pace calculation: {missing_pairs}")
            
        return df_sim

    df_sim = add_stly_otb(df_sim)
    df_sim = add_pace(df_sim)
    return df_sim


def cleanup_sim(df_sim):
    # Check which columns exist before converting
    columns_to_convert = {
        "RemSupply": float,
        "RemSupply_STLY": float,
        "Realized_Cxls": float,
        "Realized_Cxls_STLY": float,
        "DaysUntilArrival": float,
        "Pace_RemSupply": float,
        "SellingPrice": lambda x: round(x, 2),
        "Pace_SellingPrice": lambda x: round(x, 2)
    }
    
    for col, converter in columns_to_convert.items():
        if col in df_sim.columns:
            try:
                df_sim[col] = df_sim[col].apply(converter)
            except Exception as e:
                print(f"Error converting column {col}: {e}")
                # Create a default column if conversion fails
                if isinstance(converter, type):
                    df_sim[col] = converter()
        else:
            print(f"WARNING: Column {col} not found during cleanup.")
            # Create missing columns with default values
            if isinstance(converter, type):
                df_sim[col] = converter()
            else:
                df_sim[col] = 0  # Default for functions like round
                
    return df_sim


def prep_demand_features(hotel_num, prelim_csv_out=None, results_csv_out="results.csv"):
    """
    Wraps several functions that read OTB historical csv files into a DataFrame (df_sim)
    and adds relevant features that will be used to model demand & recommend pricing.

    Parameters:
        - hotel_num (int, required): 1 or 2
        - prelim_csv_out (str, optional): output the pre-processed dataFrame (raw) to this filepath.
        - results_csv_out (str, optional): Save resulting data as csv with given filepath.

    Returns
        - df_sim
    """
    assert hotel_num in (1, 2), ValueError("hotel_num must be either 1 or 2 (int).")
    if hotel_num == 1:
        capacity = H1_CAPACITY
        df_dbd = H1_DBD
    else:
        capacity = H2_CAPACITY
        df_dbd = H2_DBD

    # combine all otb files into one DataFrame
    # and add in features (LYA, actuals, etc.)

    df_sim = combine_files(hotel_num, SIM_AOD, prelim_csv_out=prelim_csv_out)
    
    # Check if df_sim is empty
    if df_sim.empty:
        print("ERROR: No data found in combined files.")
        return pd.DataFrame()
        
    # print(f"df_sim shape before feature extraction: {df_sim.shape}")
    
    df_sim = extract_features(df_sim, df_dbd, capacity)
    df_sim = merge_stly(df_sim)
    df_sim = cleanup_sim(df_sim)

    print(f"df_sim shape after processing: {df_sim.shape}")
    
    # Check if trash_can columns exist before dropping
    columns_to_drop = [col for col in trash_can if col in df_sim.columns]
    if columns_to_drop:
        df_sim.drop(columns=columns_to_drop, inplace=True)
    else:
        print("WARNING: No trash_can columns found to drop.")
        
    df_sim.fillna(0, inplace=True)
    
    if results_csv_out is not None:
        df_sim.to_csv(results_csv_out)
        print(f"'{results_csv_out}' file saved.")
    print(df_sim)
    return df_sim.dropna()

if(__name__ == "__main__"):
    # example usage      
    hotel_num = 2
    df_sim = prep_demand_features(
        hotel_num
    )
    print(df_sim.head())