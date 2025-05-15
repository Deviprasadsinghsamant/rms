import os
import pandas as pd
import numpy as np
from dbds import generate_hotel_dfs

def init_pickle_files():
    """Initialize all required pickle files"""
    
    # Create pickle directory if it doesn't exist
    pickle_dir = "./pickle"
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)
        print(f"Created directory {pickle_dir}")

    # Generate hotel data frames if they don't exist
    pickle_files = {
        "h1_res.pick": ("../data/H1.csv", 187),
        "h2_res.pick": ("../data/H2.csv", 226) 
    }

    for pickle_file, (csv_file, capacity) in pickle_files.items():
        pickle_path = os.path.join(pickle_dir, pickle_file)
        if not os.path.exists(pickle_path):
            print(f"Generating {pickle_file}...")
            df_res, df_dbd = generate_hotel_dfs(csv_file, capacity=capacity)
            
            # Save both reservation and day-by-day files
            df_res.to_pickle(pickle_path)
            dbd_file = pickle_file.replace("res", "dbd")
            df_dbd.to_pickle(os.path.join(pickle_dir, dbd_file))
            
            print(f"Saved {pickle_file} and {dbd_file}")
        else:
            print(f"{pickle_file} already exists")

    # Generate placeholder files for model training if needed
    model_files = {
        "X1_cxl.pick": (100, 5),
        "X2_cxl.pick": (100, 5),
        "y1_cxl.pick": (100,),
        "y2_cxl.pick": (100,)
    }

    for model_file, dims in model_files.items():
        model_path = os.path.join(pickle_dir, model_file)
        if not os.path.exists(model_path):
            print(f"Creating placeholder {model_file}...")
            if model_file.startswith("X"):
                data = pd.DataFrame(
                    np.random.rand(*dims), 
                    columns=[f"Feature_{i}" for i in range(dims[1])]
                )
            else:
                data = pd.Series(np.random.randint(0, 2, size=dims[0]))
            data.to_pickle(model_path)
            print(f"Saved {model_file}")
        else:
            print(f"{model_file} already exists")

if __name__ == "__main__":
    init_pickle_files()
