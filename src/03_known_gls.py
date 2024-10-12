# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from rich import print

# %%
def plot_gls(df):
    # Plot the known GLs
    def type2color(x):
        return np.where(x == 'galaxy', 'b', 'r')
    
    plt.figure(figsize=(10, 5))
    plt.subplot(111, projection='aitoff')
    plt.scatter(np.deg2rad(df['RA [deg]']).apply(lambda x: x - np.pi), 
                np.deg2rad(df['DEC [deg]']),
                s=2,
                c=type2color(df['type']),
                )
    galaxy = mpatches.Patch(color='b', label='Galaxy')
    claster = mpatches.Patch(color='r', label='Claster')
    plt.legend(handles=[galaxy, claster])
    plt.grid(True)
    plt.show()

def get_jpas():
    # Get the JPAS data
    df = pd.read_csv('../data/merged3.csv', usecols=['ra', 'dec'])
    
    return df

# %%

if __name__ == "__main__":
    # Plot the known GLs
    know_gls = pd.read_csv('../data/2024-10-11T11-47_export.csv')
    # plot_gls(know_gls)
    
    df_jpas = get_jpas()
    max_ra = df_jpas['ra'].max()
    min_ra = df_jpas['ra'].min()
    max_dec = df_jpas['dec'].max()
    min_dec = df_jpas['dec'].min()
    
    cond = (know_gls['RA [deg]'] > min_ra) & (know_gls['RA [deg]'] < max_ra) & (know_gls['DEC [deg]'] > min_dec) & (know_gls['DEC [deg]'] < max_dec)
    print(know_gls[cond])

# %%



