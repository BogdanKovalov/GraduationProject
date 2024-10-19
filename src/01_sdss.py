# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astroquery.sdss import SDSS

# %%
jpas = pd.read_csv('../data/merged3.csv')

# %%
cond_spec = jpas['specObjID'] > 0
jpas_spec = jpas[cond_spec]
jpas_spec
# %%
jpas_spec['specObjID'].to_csv('../data/specObjID.csv', index=False)

# %%
df = pd.read_csv('../data/sdss_info.csv', comment='#')
cond_spec = df['specObjID'] > 0
df_spec = df[cond_spec]
df_spec
# %%
df_spec.value_counts(['class', 'subClass'])
# %%
