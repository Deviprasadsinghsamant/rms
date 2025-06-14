"""
ON-THE-BOOKS DATA SERIALIZATION SCRIPTS - PART 1 OF 2

(REQUIRED TO REPRODUCE LOCALLY)

INTRODUCTION
-------

This script capture future-looking on the books (OTB)
data as_of every date in our sample that has one year of history and serializes them for
later ease of use using Pickle. This means the first year of data cannot be used
directly in the output, but data from that year is still needed for STLY (same-time last year) features.

This script is extremely CPU intensive and should not be attempted on a personal computer
(unless you have state of the art hardware). Hyperthreading should be enabled.

This script generates nearly 1000 Pickle (.pick) files containing on the books data 
as of each date from 2015-07-01 to 2017-08-31 for both H1 and H2. 

The subsequent save_sims_2.py script will pull STLY data from 2017-08-01 to
2018-08-01 (hence the need for OTB data from 2015-07-01 - 2016-08-31)

INSTRUCTIONS
------
Upon initial setup, after saving df_res and df_dbd to pickle files*, run this script
from your terminal.
    - *df_res and df_sim are generated from the 'dbds.py' script. Replace 'df' with 'h<n>';
      replace '<n>' with the hotel number. 

Ensure the global params are as follows: 
FOLDER = "./sims/pickle/"
START = datetime.date(2015, 8, 1)
STOP = datetime.date(2017, 8, 1)
PULL_EXTENDED = False
        

Execute from the command line, like so:
    > python3 save_sims.py
"""

from collections import defaultdict
import pandas as pd
import numpy as np
import datetime
from sim import generate_simulation
import time
import os

FOLDER = "./sims2/"  # Changed from string to path
# Ensure folder exists
if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)

START = datetime.date(2015, 8, 1)
STOP = datetime.date(2017, 8, 1)
DATE_FMT = "%Y-%m-%d"

# H1_RES = pd.read_pickle(r"C:\Users\Quotus\Desktop\rms002\rms_init\hotel-revman-system\code\pickle\h1_res.pick")
# H2_RES = pd.read_pickle(r"C:\Users\Quotus\Desktop\rms002\rms_init\hotel-revman-system\code\pickle\h2_res.pick")
# H1_DBD = pd.read_pickle(r"C:\Users\Quotus\Desktop\rms002\rms_init\hotel-revman-system\code\pickle\h1_dbd.pick")
# H2_DBD = pd.read_pickle(r"C:\Users\Quotus\Desktop\rms002\rms_init\hotel-revman-system\code\pickle\h2_dbd.pick")

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Construct the relative path to the pickle files
pickle_dir = os.path.join(script_dir, "pickle")

H1_RES = pd.read_pickle(os.path.join(pickle_dir, "h1_res.pick"))
H2_RES = pd.read_pickle(os.path.join(pickle_dir, "h2_res.pick"))
H1_DBD = pd.read_pickle(os.path.join(pickle_dir, "h1_dbd.pick"))
H2_DBD = pd.read_pickle(os.path.join(pickle_dir, "h2_dbd.pick"))



def save_sim_records(df_dbd, df_res, hotel_num, skip_existing=False, pull_extended=False):
    try:
        all_dates = [
            datetime.datetime.strftime(START + datetime.timedelta(days=x), format=DATE_FMT)
            for x in range((STOP - START).days + 1)
        ]

        counter = 1
        for as_of_date in all_dates:
            out_file = os.path.join(FOLDER, f"h{str(hotel_num)}_sim_{as_of_date}.pick")
            if skip_existing and os.path.exists(out_file):
                print(f"Skipping existing file {out_file}")
                continue
                
            df_sim = generate_simulation(
                df_dbd,
                as_of_date, 
                hotel_num,
                df_res,
                confusion=False,
                verbose=0,
            )
            df_sim.to_pickle(out_file)
            counter += 1
            print(f"Saved file {out_file}.")
        return True
    except Exception as e:
        print(f"Error saving simulation records: {str(e)}")
        return False


def save_historical_OTB(h1_dbd, h1_res, h2_dbd, h2_res):
    try:
        print("Starting hotel 1...")
        if not save_sim_records(h1_dbd, h1_res, 1):
            raise Exception("Failed to save Hotel 1 records")

        print(
            f"Finished retrieving historical OTB records for Hotel 1\n",
            "Sleeping ten seconds for CPU health..."
        )

        print("Starting hotel 2...")
        if not save_sim_records(h2_dbd, h2_res, 2):
            raise Exception("Failed to save Hotel 2 records")
            
        print("All files saved.")
        return True
    except Exception as e:
        print(f"Error saving historical OTB data: {str(e)}")
        return False


if __name__ == "__main__":
    save_historical_OTB(H1_DBD, H1_RES, H2_DBD, H2_RES)