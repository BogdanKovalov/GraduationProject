# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import scipy.signal as signal
from astropy.io import fits
import os

# %%
def load_jpas():
    # Load the JPAS data
    jpas = pd.read_csv('../data/merged3.csv')
    return jpas

def load_jpas_filters(slice=np.s_[0:-4], isshow=False):
    # Load the JPAS filters
    filters = pd.read_csv('../data/jpas_filters.csv', comment='#')
    filters = filters[slice]
    if isshow:
        # Plot the JPAS filters
        fig, ax = plt.subplots()
        ax.bar(x = filters['wavelength'],
               height = 1,
               width = filters['width'],
               align='center',
            #    color=plt.cm.tab20(np.arange(len(filters)) % 20),
               color=filters['color_representation'],
               edgecolor='black',
               alpha=0.7,
               lw=0.5,
               label='JPAS filters',
               )
        ax.set_xlabel('Wavelength [Angstrom]')
        ax.set_ylabel('Transmission')
        ax.legend()
        plt.show()
        
    return filters

def val2array(val):
    return np.array(val.split(' '), dtype=float)

def load_sdss_spectum(i, sp_obj_id):
    file_name = f"{i}_{sp_obj_id}.fits"
    print(f"Loading {file_name}")
    hdul = fits.open(f'../data/sdss_spectra/{i}_{sp_obj_id}.fits')
    data = hdul[1].data
    
    return data
    

def plot_jpas_spec(row, jpas_filters, slice=np.s_[0:-4], issave=False):
    # JPAS fluxes
    # print(f"RA, DEC: {row['ra']:.6f} {row['dec']:.6f}")
    # fluxes = val2array(row['FLUX_AUTO'])
    print("JPAS fluxes")
    fluxes = val2array(row['FLUX_APER_4_0'])
    fluxes_err = val2array(row['FLUX_RELERR_APER_4_0'])[slice]
    fluxes_err = np.where(fluxes_err < 0, 1e-5, fluxes_err)
    
    k = 1.0
    x = jpas_filters['wavelength']
    # y = fluxes[slice] / jpas_filters['width'] * k 
    y = fluxes[slice] * k / 100
    
    try:
        sdss_data = load_sdss_spectum(row['jpas_idx'], row['specObjID'])
        wavelenght_sdss = 10**sdss_data['loglam']
    except:
        return None
    
    plt.figure()
    plt.errorbar(x, y,
                yerr=fluxes_err / 10 * k,
                xerr=jpas_filters['width'] / 2,
                fmt='.', 
                color='black', 
                lw=0.5,
                zorder=5,
                )
    
    plt.scatter(x, y,
            color=jpas_filters['color_representation'], 
            s=20, zorder=6,
            label='JPAS fluxes')
    
    
    plt.plot(wavelenght_sdss, sdss_data['flux'],
             alpha=0.2, color='black', 
             zorder=3,
             label='SDSS fluxes')
    plt.plot(wavelenght_sdss, sdss_data['model'], 
             zorder=4,
             label='SDSS model')
    
    plt.xlabel('Wavelength [Angstrom]')
    plt.ylabel(r'Flux [$10^{-17}$ erg/s/$cm^2$/Angstrom]')
    plt.suptitle(f"JPAS AND SDSS spectra JPAS Idx:{row['jpas_idx']}")
    plt.grid(alpha=0.5)
    plt.ylim(min(sdss_data['model'].min(), y.min()) * 0.5, 
             max(sdss_data['model'].max(), y.max()) * 1.1
    )
    plt.legend()
    
    plt.tight_layout()
    
    if issave:
        if (os.path.exists('../images/spectra/jpas_sdss_{idx}.png') == False):
            os.makedirs('../images/spectra/jpas_sdss_{idx}', exist_ok=True)

        plt.savefig(f"../images/spectra/jpas_sdss_{idx}.png",
                    edgecolor='white')
        plt.close()
    else:
        plt.show()
    
def resample_spectrum(sdss_data, jpas_data, wavelenght_sdss, wavelenght_jpas):
    
    min_spectrum = min(wavelenght_sdss.min(), wavelenght_jpas.min())
    max_spectrum = max(wavelenght_sdss.max(), wavelenght_jpas.max())
    num_points = (max_spectrum - min_spectrum) / 50

    x = 10**sdss_data['loglam'].astype(float)
    y_sdss = sdss_data['flux'].astype(float)

    interp_function = interpolate.interp1d(x, y_sdss)
    x = np.arange(x.min(), x.max(), 1)
    y_sdss = interp_function(x)

    x = np.insert(x, 0, min_spectrum)
    y_sdss = np.insert(y_sdss, 0, y_sdss[:5].mean())

    cond = x < max_spectrum
    x = x[cond]
    y_sdss = y_sdss[cond]
    
    x = np.append(x, max_spectrum)
    y_sdss = np.append(y_sdss, y_sdss[-5:].mean())

    y_resampled_sdss = signal.resample(y_sdss, int(num_points) + 1)
    x_resampled = np.linspace(min_spectrum, max_spectrum, int(num_points) + 1)

    x = np.array(wavelenght_jpas)
    y_jpas = np.array(jpas_data)

    cond = x > min_spectrum
    x = x[cond]
    np.insert(x, 0, min_spectrum)
    
    x = np.append(x, max_spectrum)
    y_jpas = y_jpas[cond]
    y_jpas = np.append(y_jpas, y_jpas[-1])

    y_resampled_jpas = signal.resample(y_jpas, int(num_points) + 1)

    plt.plot(x_resampled, y_resampled_sdss, marker='o', label='Resampled SDSS')
    plt.plot(x_resampled, y_resampled_jpas / 100 - 1, marker='o', label='Resampled JPAS')
    plt.show()



# %%
if __name__ == '__main__':
    
    # Load the JPAS data
    df_jpas = load_jpas()
    
    # Selection of the SDSS spectrum objects
    cond_spec = df_jpas['specObjID'] > 0
    df_jpas_spec = df_jpas[cond_spec]
    
    # JPAS filters
    jpas_slice = np.s_[0:-4]
    df_jpas_filters = load_jpas_filters(slice=jpas_slice, 
                                        isshow=True)
    
    sdss_data = load_sdss_spectum(df_jpas_spec['jpas_idx'].iloc[0], df_jpas_spec['specObjID'].iloc[0])
    print(sdss_data, sdss_data)
    for i, (idx, row) in enumerate(df_jpas_spec[1:100].iterrows()):
        print(i,idx)
        # if i < 10:
        #     continue
        # JPAS fluxes
        # sdss_data = load_sdss_spectum(row['jpas_idx'], row['specObjID'])
        plot_jpas_spec(row, df_jpas_filters,
                       slice=jpas_slice, issave=False)
        
    
    
# %%
df_jpas = load_jpas()
print(df_jpas['FLUX_APER_4_0'].iloc[0])
sdss_data = load_sdss_spectum(df_jpas['jpas_idx'].iloc[0], df_jpas['specObjID'].iloc[0])
wavelenght_sdss = 10**sdss_data['loglam']
wavelenght_jpas = df_jpas_filters['wavelength'][:-4].astype(float)
resample_spectrum(sdss_data, val2array(df_jpas['FLUX_APER_4_0'].iloc[0])[:-8], wavelenght_sdss, wavelenght_jpas)
#df_resampled = pd.DataFrame(
#    columns=['wavelenght', 'jpas_flux', 'jpas_flux_err', 
#             'sdss_flux', 'sdss_flux_err', 'sdss_model'])
# %%
