# %%
from pathlib import Path
import pandas as pd

# %%
root = Path(r'C:/Users/cnolan/OneDrive - UNSW/ACAN2023/Modules/Week 3_Conditioning Module/~RAW Data')
fns = pd.DataFrame([str(fn) for fn in root.glob('**/Backup*.txt')], columns=['path'])
fns.to_csv('medpc_mapping.csv', index=False)

# %%
from behapy import medpc

# %%
event_map = {
        1: "LP",
        3: "Rew",
        4: "Mag"
    }
medpc.medpc2bids('medpc_mapping.csv', '../temp', 'Z', 'Y', event_map)

# %%
