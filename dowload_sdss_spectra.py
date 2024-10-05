# %%
import pandas as pd
from rich import print
import os

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.sdss import SDSS

# %%
df_merged = pd.read_csv('./data/merged2.csv')
cond_sp = df_merged["specObjID"] != 0
df_spectra = df_merged[cond_sp]

for i, row in df_spectra.iterrows():
    print(f"i: {i}")
    specObjID = row["specObjID"]
    jpasID = row["jpas_idx"]
    
    file_name = f'./data/sdss_spectra/{jpasID}_{specObjID}.fits'
    if os.path.exists(file_name):
        print(f"File already exists: {file_name}")
        continue
    
    coords_sp = SkyCoord(ra=df_spectra.loc[i, 'ra'], 
                         dec=df_spectra.loc[i, 'dec'], 
                         unit=(u.deg, u.deg), frame='icrs')
    
    try:
        query = SDSS.query_region(coords_sp, radius=2*u.arcsec,
                    spectro=True)
        print(query)
        mjd = query['mjd'][-1]
        print(f"jpasID: {jpasID}, specObjID: {specObjID}")
        result = SDSS.get_spectra(
            matches=query[-1:],
            # coordinates=coords_sp,
            # mjd=mjd,
            # radius=2*u.arcsec,
        )
    except Exception as e:
        print(f"Error: {e}")
        continue
    else:
        print(f"result: {result}")
        print(f"Saving file: {file_name}")
        result[0].writeto(file_name, overwrite=True)
    
    # break
        

# %%
