import pandas as pd
import numpy as np
import tables
import os

METERS = {
    'aggregate': 1,
    'fridge': 12,
    'dishwasher': 6,
    'microwave': 13
}

# Dynamic window configurations per appliance
APP_CONFIG = {
    'fridge': {'window': 599, 'step': 6},
    'dishwasher': {'window': 599, 'step': 6},
    'microwave': {'window': 99, 'step': 2} 
}

def extract_sliding_windows(series, window_size, step=1):
    num_windows = (series.size - window_size) // step + 1
    shape = (num_windows, window_size)
    strides = (series.strides[0] * step, series.strides[0])
    return np.lib.stride_tricks.as_strided(series, shape=shape, strides=strides)

def load_meter_data(h5_file, building, meter_id, sample_rate):
    node = h5_file.get_node(f"/building{building}/elec/meter{meter_id}/table")
    data = node.read()
    timestamps = pd.to_datetime(data["index"])
    values = np.array([row[0] for row in data["values_block_0"]], dtype=np.float32)
    series = pd.Series(values, index=timestamps)
    return series.resample(sample_rate).mean().ffill().fillna(0)

def load_and_preprocess(h5_path, building=1, sample_rate='10s'):
    print(f"Loading data from {h5_path}...")
    h5 = tables.open_file(h5_path, mode='r')
    df_dict = {}

    for name, meter_id in METERS.items():
        try:
            df_dict[name] = load_meter_data(h5, building, meter_id, sample_rate)
        except Exception as e:
            print(f"Failed loading {name}: {e}")
            h5.close()
            raise
    h5.close()

    df_merged = pd.concat(df_dict, axis=1).dropna()
    print(f"Total synchronized samples: {len(df_merged)}")

    agg_max = max(df_merged["aggregate"].max(), 1.0)
    df_norm = pd.DataFrame(index=df_merged.index)
    df_norm["aggregate"] = df_merged["aggregate"] / agg_max

    scales_dict = {"aggregate": agg_max}
    X_dict, Y_dict = {}, {}
    
    for appliance, config in APP_CONFIG.items():
        print(f"Processing {appliance} (Window: {config['window']}, Step: {config['step']})...")
        app_max = max(df_merged[appliance].max(), 1.0)
        scales_dict[appliance] = app_max
        df_norm[appliance] = df_merged[appliance] / app_max
        
        w_size = config['window']
        step = config['step']
        
        X_agg = extract_sliding_windows(df_norm["aggregate"].values, w_size, step=step)
        X_dict[appliance] = X_agg
        
        midpoint = w_size // 2
        num_windows = X_agg.shape[0]
        indices = np.arange(num_windows) * step + midpoint
        
        Y_dict[appliance] = df_norm[appliance].values[indices]

    return X_dict, Y_dict, scales_dict

if __name__ == "__main__":
    h5_file = "ukdale.h5"
    if not os.path.exists(h5_file):
        print(f"File not found: {h5_file}")
        exit()

    X_dict, Y_dict, scales = load_and_preprocess(h5_file, building=1)

    np.savez_compressed(
        "processed_ukdale.npz",
        X_fridge=X_dict["fridge"],
        X_dishwasher=X_dict["dishwasher"],
        X_microwave=X_dict["microwave"],
        Y_fridge=Y_dict["fridge"],
        Y_dishwasher=Y_dict["dishwasher"],
        Y_microwave=Y_dict["microwave"],
        scales=np.array([scales], dtype=object)
    )
    print("Preprocessing complete with multi-resolution mappings.")