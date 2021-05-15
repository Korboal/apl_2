import numpy as np
import datetime
from calculating_v7_1 import get_ids_and_z, generate_IDs_to_analyse, get_HST_IDs, get_no_HST_IDs, \
    get_noisy_IDs, load_file_as_integers, new_fitting_with_image


version_number = 7_1            # only for textfile saving
extra_filename = "_calc_"       # only for textfile saving
filename_all_hst_val_only = "output_textfiles/" + str(datetime.datetime.now()).replace(':', '_') + "_values_v" + str(version_number) + extra_filename + ".txt"

def main():
    Lya_fitting = True     # Fit one image of Lya
    UV_fitting = False       # Fit one image of UV
    image_info = False       # Whether to show image info while fitting one image
    image_showing = False   # Whether to show output fits + residuals
    auto_save_image = False  # Whether to save output fits + residuals

    number_of_slices = 1000     # For uncertainty estimates; how many slices. More -> longer time. 1000 is default
    calculate_uncertainties = False  # calculate all values, uncertainties and separation at once and save for Topcat

    # Load in files and see which ones are isolated. From there create two textfiles: ones with HST counterpart and ones without

    generate_IDs_to_analyse("analysis_of_ids/image_analysis_v2.txt")    # generates IDs to analyse from this file. Format: ID, isolated (1 is isolated), centered (ignored atm), 3rd panel OK (ignored atm), HST exists (1 exists, 0 does not)
    IDs_hst = get_HST_IDs()         # basically rewrite next two lines if you want to load IDs in your own manner
    IDs_no_hst = get_no_HST_IDs()

    # Files loading: data new 15.03.2021

    path_Lya = "data_new/muse_10_as/NBLya_10as_MID"         # start name of image file, with ID after that name
    path_UV = "data_new/hst_4_as/hstUVmasked_4as_MID"       # start name of image file, with ID after that name
    path_txt_file = "data_new/APL2_mxdf_LyaZCONF23_IdRidRaDecZRaDec.txt"    # Parameters of images. Format: each line is numbers Id, RID, Ra Lya, Dec Lya, Z, Ra Uv, Dec UV
    path_PSF_param = "data_new/APL2_mxdf_PSFparameters.txt"                 # PSF parameters. Format: ID, Alpha, Beta, Gamma
    path_noisy_IDs = "analysis_of_ids/noisy_image__ID_cenx_ceny_tol_lya.txt"    # Images that are noisy and need to have mask applied. Format: ID, center_x (in pixels), center_y (in pixels), tolerance (in pixels, within tolerance no masking), lya (1 is lya, 0 is uv)
    path_high_var_IDs = "analysis_of_ids/high_var__ID_vartol_lya.txt"           # Images that have high variance. Format: ID, variance tolerance (int; variance with var_mean + var_tol * var_st.dev are masked), lya (1 is lya, 0 is uv)
    path_fitting_method_IDs = "analysis_of_ids/fitting_method__ID_method_lya.txt"   # File with fitting methods. Format: ID, integer for method (e.g. 0 is leastsq, 1 is least_squares), lya (1 is lya, 0 is uv)
    path_end = ".fits"                                       # image extension

    values = np.loadtxt(path_txt_file)  # all IDs that will be analysed
    IDs = values[:, 0].astype(int)      # next 7 arrays not used anywhere, purely for your own understanding
    Rid = values[:, 1].astype(int)
    Ra_Lya = values[:, 2]
    Dec_Lya = values[:, 3]
    z_s = values[:, 4]
    Ra_UV = values[:, 5]
    Dec_UV = values[:, 6]

    psf_values = np.loadtxt(path_PSF_param)     # PSF parameters for all values. you need them for ALL
    IDs_psf = psf_values[:, 0].astype(int)  # next 4 arrays not used anywhere, purely for your own understanding
    alpha_psf = psf_values[:, 1]        # Lya
    beta_psf = psf_values[:, 2]         # Lya
    sigma_psf = psf_values[:, 3]        # UV

    noisy_IDs = get_noisy_IDs(path_noisy_IDs)
    high_var_IDs = load_file_as_integers(path_high_var_IDs)
    fitting_methods_IDs = load_file_as_integers(path_fitting_method_IDs)
    lya_fitting_methods = fitting_methods_IDs[fitting_methods_IDs[:, 2] == 1]
    uv_fitting_methods = fitting_methods_IDs[fitting_methods_IDs[:, 2] == 0]

    IDs_to_find = [53, 68, 149, 153, 180, 1817, 6666, 6700, 7089, 7091] # if want to only use specific IDs from the full sample
    IDs_to_find = IDs_hst           # if only IDs that have UV
    IDs_to_find = IDs_no_hst        # if IDs that ONLY have Lya

    # noinspection PyRedeclaration      #start at 417
    IDs, Rid, Ra_Lya, Dec_Lya, z_s, Ra_UV, Dec_UV, IDs_psf, alpha_psf, beta_psf, sigma_psf = get_ids_and_z(values, psf_values, IDs_to_find, start_at_ID=0)  # otherwise comment this line
    psf_array = np.array([IDs_psf, alpha_psf, beta_psf, sigma_psf])

    # Create actual paths

    paths_Lya = np.empty(len(IDs), dtype='object')
    paths_UV = np.empty(len(IDs), dtype='object')

    for i in range(len(IDs)):
        paths_Lya[i] = path_Lya + str(IDs[i]) + path_end
        paths_UV[i] = path_UV + str(IDs[i]) + path_end

    # Fitting different images

    new_fitting_with_image(paths_Lya, paths_UV, psf_array, IDs, z_s, Ra_Lya, Dec_Lya, Ra_UV, Dec_UV,
                           filename_all_hst_val_only, number_of_slices, Lya_fitting, UV_fitting,
                           calculate_uncertainties, image_info, image_showing, auto_save_image,
                           noisy_IDs=noisy_IDs, high_var_IDs=high_var_IDs,
                           fitting_method_lya=lya_fitting_methods, fitting_method_uv=uv_fitting_methods)

if __name__ == "__main__":      # required if using Dask; works either way
    main()