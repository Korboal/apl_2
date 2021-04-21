import numpy as np
import astropy.units as u
from mpdaf.obj import Image, Cube
from gauss_fitting_v6_3 import gauss2d_fitting_no_plot, flatten_xy_err, plot_image, gauss2d_fitting_no_plot_params, \
    convert_params_to_variables, gauss2d_fitting_no_plot_params_img, convert_data_to_odd_axes
import datetime
from astropy.cosmology import FlatLambdaCDM
from dask.distributed import Client, progress
import matplotlib
import scipy

filepath_HST_IDs = "isolated_HST_IDs.txt"
filepath_no_HST_IDs = "isolated_no_HST_IDs.txt"
masking_value = np.nan
default_std_var = 3
threshold_for_workers = 20
workers = 12  # should be the same as cores
threads_per_worker = 1  # seemed to work best with 1
debug_mode = False

def new_fitting_with_image(paths_Lya, paths_UV, psf_array, IDs, z_s, Ra_lya, Dec_lya, Ra_uv, Dec_uv, filename_val_only, number_of_slices, lya_fit, uv_fit,
                           uncertainty_calc,  image_info, image_showing, save_image,
                           noisy_IDs=np.array([[0]]), high_var_IDs=np.array([[0]]), model_lya=np.array([[0]]), model_uv=np.array([[0]]),
                           fitting_method_lya=np.array([[0]]), fitting_method_uv=np.array([[0]])):
    """
    Calculates and fits values and uncertainty for center, sigma, ellipticity and separation. One function that does everything
    :param paths_Lya: 1D array the paths to the Lya images
    :param paths_UV: 1D array the path to the UV images
    :param psf_array: 2D array with PSF parameters. psf_array[j] gives 1D array of specific values. j=0: ID, j=1: alpha, j=2: beta,
    j=3: sigma of PSF. psf_array[j][i] gives i-th 1D array with j-th parameter for the specific image.
    :param IDs: 1D array the IDs to use
    :param z_s: 1D array the redshift z of the images
    :param Ra_lya: 1D array the given Ra Lya value
    :param Dec_lya: 1D array the given Dec Lya value
    :param Ra_uv: 1D array the given Ra UV value
    :param Dec_uv: 1D array the given Dec UV value
    :param filename_val_only: String - Path + filename where to save values
    :param number_of_slices: integer - number of slices to do uncertainty calculations
    :param lya_fit: True/False if need to do Lya fit
    :param uv_fit: True/False if need to do UV fit
    :param uncertainty_calc: True/False whether to do uncertainty calculations using number_of_slices
    :param image_info: True/False whether to show image information
    :param image_showing: True/False whether to show fitted image
    :param save_image: True/False whether to save image into an output folder
    :param noisy_IDs: optional, 2D array with IDs that need to be masked. Format: ID, center_x (in pixels), center_y
    (in pixels), tolerance (in pixels, within tolerance no masking), lya (1 is lya, 0 is uv)
    :param high_var_IDs: optional, 2D array with IDs that need high variance pixels to be masked. Format: ID,
    variance tolerance (int; variance with var_mean + var_tol * var_st.dev are masked), lya (1 is lya, 0 is uv)
    :param model_lya: optional, 2D array with models. Format: ID (int), model (int; 0 = gaussian2D).
    :param model_uv: optional, 2D array with models. Format: ID (int), model (int; 0 = gaussian2D).
    :param fitting_method_lya: optional, 2D array with fitting methods. Format: ID, integer for method (e.g. 0 is
    leastsq, 1 is least_squares), lya (1 is lya, 0 is uv)
    :param fitting_method_uv: optional, 2D array with fitting methods. Format: ID, integer for method (e.g. 0 is
    leastsq, 1 is least_squares), lya (1 is lya, 0 is uv)
    :return: Returns nothing, but appends filenames with values. Text saved in following order (if only Lya, skips
    UV and separation parts; saved for use in TOPCAT program, just load table as ASCII):
    #ID z amp_l d_amp_l cen_x_l d_cen_x_l cen_y_l d_cen_y_l sig_x_l d_sig_x_l sig_y_l d_sig_y_l ll_l d_ell_l
    amp_u d_amp_u cen_x_u d_cen_x_u cen_y_u d_cen_y_u sig_x_u d_sig_x_u sig_y_u d_sig_y_u ell_u d_ell_u
    ang_sep d_ang_sep kpc_sep d_kpc_sep lya_ra_fit lya_dec_fit lya_ra lya_dec uv_ra_fit uv_dec_fit uv_ra uv_dec
    """

    if uncertainty_calc and number_of_slices >= threshold_for_workers and not debug_mode:
        matplotlib.use('Agg')
        print("Preparing workers")
        global client
        client = Client(threads_per_worker=threads_per_worker, n_workers=workers)
        print("Worker preparation complete")

    if uncertainty_calc and lya_fit and uv_fit:     # prepares text file to save values/uncertainties
        save_in_txt(["#ID\tz\tamp_l\td_amp_l\tcen_x_l\td_cen_x_l\tcen_y_l\td_cen_y_l\tsig_x_l\td_sig_x_l\tsig_y_l\td_sig_y_l\t"
                        "ell_l\td_ell_l"
                        "\tamp_u\td_amp_u\tcen_x_u\td_cen_x_u\tcen_y_u\td_cen_y_u\tsig_x_u\td_sig_x_u\tsig_y_u\td_sig_y_u\t"
                        "ell_u\td_ell_u"
                        "\tang_sep\td_ang_sep\tkpc_sep\td_kpc_sep\tlya_ra_fit\tlya_dec_fit\tlya_ra\tlya_dec\tuv_ra_fit\t"
                        "uv_dec_fit\tuv_ra\tuv_dec\tsn_l\tsn_u"], filename_val_only)

    if uncertainty_calc and lya_fit and not uv_fit: # prepares text file to save values/uncertainties, if no HST
        save_in_txt(
            ["#ID\tz\tamp_l\td_amp_l\tcen_x_l\td_cen_x_l\tcen_y_l\td_cen_y_l\tsig_x_l\td_sig_x_l\tsig_y_l\td_sig_y_l\t"
             "ell_l\td_ell_l"
             "\tlya_ra_fit\tlya_dec_fit\tlya_ra\tlya_dec\tsn_l"], filename_val_only)

    for i in range(len(IDs)):   # goes through each ID
        if lya_fit:     # Lya fit
            image_lya = Image(paths_Lya[i])
            print("Lya image of", IDs[i])
            if image_info:
                print(image_lya.info())

            amplitude_lya, center_deg_lya, center_x_lya, center_y_lya, ellip_lya, sigma_x_lya, sigma_y_lya, s_n_r_lya = fit_image(
                IDs[i], fitting_method_lya, high_var_IDs, image_lya, image_showing, model_lya, noisy_IDs, psf_array[:, i],
                save_image, True)

            if uncertainty_calc:    # uncertainty calculations for Lya fit
                spec_model_lya = get_model_name(IDs[i], model_lya)
                spec_fitting_method_lya = get_fitting_method(IDs[i], fitting_method_lya)

                prior_arr_lya = get_prior_array(amplitude_lya, center_x_lya, center_y_lya, sigma_x_lya, sigma_y_lya)

                uncertainties_lya = get_uncertainties(IDs[i], image_lya, psf_array[:, i], number_of_slices,
                                                      noisy_IDs,
                                                      high_var_IDs, spec_model_lya, spec_fitting_method_lya, True, priors=prior_arr_lya)
                d_amp_l, d_cen_x_l, d_cen_y_l, d_sig_x_l, d_sig_y_l, d_ellip_l, centers_deg_l = convert_uncertainties_to_variables(
                    uncertainties_lya)

                if not uv_fit:  # saves uncertainties + values if no HST
                    print("Full calculation done for Lya", IDs[i])
                    save_in_txt_topcat([IDs[i], z_s[i], amplitude_lya, d_amp_l, center_x_lya, d_cen_x_l,
                                        center_y_lya, d_cen_y_l, sigma_x_lya, d_sig_x_l, sigma_y_lya,
                                        d_sig_y_l,
                                        ellip_lya, d_ellip_l,
                                        center_deg_lya[1], center_deg_lya[0], Ra_lya[i], Dec_lya[i], s_n_r_lya],
                                       filename_val_only)

        if uv_fit:  # UV fit
            image_uv = Image(paths_UV[i])
            print("UV image of", IDs[i])
            if image_info:
                print(image_uv.info())

            amplitude_uv, center_deg_uv, center_x_uv, center_y_uv, ellip_uv, sigma_x_uv, sigma_y_uv, s_n_r_uv = fit_image(
                IDs[i], fitting_method_uv, high_var_IDs, image_uv, image_showing, model_uv, noisy_IDs, psf_array[:, i],
                save_image, False)

            if uncertainty_calc:    # uncertainty calculations for UV
                spec_model_uv = get_model_name(IDs[i], model_uv)
                spec_fitting_method_uv = get_fitting_method(IDs[i], fitting_method_uv)

                prior_arr_uv = get_prior_array(amplitude_uv, center_x_uv, center_y_uv, sigma_x_uv, sigma_y_uv)

                image_uv_2, spec_noisy_ID, prior_arr_uv = reduce_image(image_uv, center_x_uv, center_y_uv, sigma_x_uv, sigma_y_uv,
                                                     noisy_IDs[noisy_IDs[:, 4] == 0], IDs[i], priors=prior_arr_uv)

                uncertainties_uv = get_uncertainties(IDs[i], image_uv_2, psf_array[:, i], number_of_slices, spec_noisy_ID,
                                                     high_var_IDs, spec_model_uv, spec_fitting_method_uv, False, priors=prior_arr_uv)
                d_amp_u, d_cen_x_u, d_cen_y_u, d_sig_x_u, d_sig_y_u, d_ellip_u, centers_deg_u = convert_uncertainties_to_variables(
                    uncertainties_uv)

        if uncertainty_calc and lya_fit and uv_fit:  # lya and hst counter-part; calculates separations
                lya_uv_ang_sep = ang_sep(center_deg_lya, center_deg_uv) * 3600  # converts degrees to arcsec
                diff_in_kpc = kpc_sep(lya_uv_ang_sep, z_s[i]) / u.kpc

                lya_uv_ang_seps = ang_sep_many(centers_deg_l, centers_deg_u) * 3600  # converts degrees to arcsec
                diff_in_kpcs = kpc_sep_many(lya_uv_ang_seps, z_s[i]) / u.kpc

                print("Full calculation done for", IDs[i], "such that separation is", lya_uv_ang_sep, "arcsec or",
                      diff_in_kpc, "kpc")
                save_in_txt_topcat([IDs[i], z_s[i], amplitude_lya, d_amp_l, center_x_lya, d_cen_x_l,
                                    center_y_lya, d_cen_y_l, sigma_x_lya, d_sig_x_l, sigma_y_lya,
                                    d_sig_y_l,
                                    ellip_lya, d_ellip_l,
                                    amplitude_uv, d_amp_u, center_x_uv, d_cen_x_u,
                                    center_y_uv, d_cen_y_u, sigma_x_uv, d_sig_x_u, sigma_y_uv,
                                    d_sig_y_u,
                                    ellip_uv, d_ellip_u,
                                    lya_uv_ang_sep, np.std(lya_uv_ang_seps), diff_in_kpc, np.std(diff_in_kpcs),
                                    center_deg_lya[1],
                                    center_deg_lya[0], Ra_lya[i], Dec_lya[i], center_deg_uv[1], center_deg_uv[0],
                                    Ra_uv[i], Dec_uv[i], s_n_r_lya, s_n_r_uv], filename_val_only)    # saves everything in text file


def get_prior_array(amplitude, center_x, center_y, sigma_x, sigma_y):
    prior_arr = np.array(
        [amplitude, 0, 0, center_x, center_x - 2 * sigma_x, center_x + 2 * sigma_x, center_y,
         center_y - 2 * sigma_y, center_y + 2 * sigma_y, sigma_x, sigma_x / 3, sigma_x * 3,
         sigma_y, sigma_y / 3, sigma_y * 3, 0])
    return prior_arr


def convert_uncertainties_to_variables(uncertainties):
    """
    Converts 3D array uncertainties into separate variables: 6 floats and 2D array
    :param uncertainties: 3D array gotten from the calculations. Format: [float amp, float cen_x, float cen_y, float
    sig_x, float sig_y, float ellip, [2D array of the centers in ra and dec]]
    :return: 6 floats and 2D array. Order: float amp, float cen_x, float cen_y, float sig_x, float sig_y, float ellip,
    [2D array of the centers in ra and dec]
    """
    d_amp = uncertainties[0]
    d_cen_x = uncertainties[1]
    d_cen_y = uncertainties[2]
    d_sig_x = uncertainties[3]
    d_sig_y = uncertainties[4]
    d_ellip = uncertainties[5]
    centers_deg = uncertainties[6]
    return d_amp, d_cen_x, d_cen_y, d_sig_x, d_sig_y, d_ellip, centers_deg

def fit_image(ID, fitting_method, high_var_IDs, image, image_showing, model, noisy_IDs, psf_array, save_image, lya):
    """
    Takes the image and inputs, decideds which model to use for fitting (default is gaussian 2D).
    :param ID: int ID of the image
    :param fitting_method: 2D array with fitting methods. Format: ID, integer for method (e.g. 0 is
    leastsq, 1 is least_squares), lya (1 is lya, 0 is uv)
    :param high_var_IDs: 2D array with IDs that need high variance pixels to be masked. Format: ID,
    variance tolerance (int; variance with var_mean + var_tol * var_st.dev are masked), lya (1 is lya, 0 is uv)
    :param image: MPDAF image
    :param image_showing: True/False whether to show image
    :param model: 2D array with models. Format: ID (int), model (int; 0 = gaussian2D).
    :param noisy_IDs: 2D array with IDs that need to be masked. Format: ID, center_x (in pixels), center_y
    (in pixels), tolerance (in pixels, within tolerance no masking), lya (1 is lya, 0 is uv)
    :param psf_array: 1D array with PSF parameters. psf_array[j] gives values for PSF for that ID. j=0: ID, j=1: alpha,
    j=2: beta, j=3: sigma of PSF
    :param save_image: True/False whether to save image into an output folder
    :param lya: True/False. True means Lya image, False means UV image
    :return: 6 floats and 1 2D array, parameters of the fit. Order: amplitude, 2D array of center_deg [dec, ra],
    center_x, center_y, ellip, sigma_x, sigma_y
    """
    fit_model = get_model_name(ID, model)   # by default returns 'gaussian2D'
    if fit_model == 'gaussian2D':
        amplitude, center_x, center_y, sigma_x, sigma_y, ellip, center_deg, sn_r = fit_image_2d_gaussian(
            image, ID, high_var_IDs, noisy_IDs, psf_array, image_showing, save_image,
            fitting_method, lya)
    elif fit_model == 'something_else':
        amplitude, center_x, center_y, sigma_x, sigma_y, ellip, center_deg = fit_image_smth_else(
            image, ID, high_var_IDs, noisy_IDs, psf_array, image_showing, save_image,
            fitting_method, lya)
    return amplitude, center_deg, center_x, center_y, ellip, sigma_x, sigma_y, sn_r


def get_model_name(ID, model_array):
    """
    Checks whether ID is in the model array and returns the name of model array. By default, returns 'gaussian2D'
    :param ID: integer ID of the object
    :param model_array: 2D array with models. Format: ID (int), model (int; 0 = gaussian2D).
    :return: string name of the model. E.g. 'gaussian2D'
    """
    if ID in model_array[:, 0]:
        index = np.where(model_array[:, 0] == ID)[0][0]
        if model_array[index, :][1] == 0:
            fit_model = 'gaussian2D'
        else:
            print("Unknown model, assuming Gaussian 2D")
            fit_model = 'gaussian2D'
    else:
        fit_model = 'gaussian2D'
    return fit_model


def get_fitting_method(ID, fitting_method):
    """
    Takes ID and 2D array of fitting methods. Checks that fitting method exists and returns its name. By default returns
    'leastsq'
    Method names:
    0: leastsq
    1: least_squares
    :param ID: int ID of object
    :param fitting_method: 2D array with fitting methods. Format: ID, integer for method (0, 1 etc)
    :return: Name of the method, e.g. 'leastsq', 'least_squares'
    """
    if ID in fitting_method[:, 0]:
        index = np.where(fitting_method[:, 0] == ID)[0][0]
        if fitting_method[index, :][1] == 0:
            method = 'leastsq'
        elif fitting_method[index, :][1] == 1:
            method = 'least_squares'
        else:
            print("Unknown method, assuming leastsq")
            method = 'leastsq'
    else:
        method = 'leastsq'
    return method


def fit_image_2d_gaussian(image, ID, high_var_IDs, noisy_IDs, psf_array, image_showing, save_image, fitting_method, lya):
    """
    Does the fitting using 2D Gaussian model
    :param image: MPDAF image
    :param ID: ID of the object
    :param high_var_IDs: 2D array with IDs that need high variance pixels to be masked. Format: ID,
    variance tolerance (int; variance with var_mean + var_tol * var_st.dev are masked), lya (1 is lya, 0 is uv)
    :param noisy_IDs: 2D array with IDs that need to be masked. Format: ID, center_x (in pixels), center_y
    (in pixels), tolerance (in pixels, within tolerance no masking), lya (1 is lya, 0 is uv)
    :param psf_array: 1D array with PSF parameters. psf_array[j] gives values for PSF for that ID. j=0: ID, j=1: alpha,
    j=2: beta, j=3: sigma of PSF
    :param image_showing: True/False whether to show image
    :param save_image: True/False whether to save image into an output folder
    :param fitting_method: 2D array with fitting methods. Format: ID, integer for method (e.g. 0 is
    leastsq, 1 is least_squares), lya (1 is lya, 0 is uv)
    :param lya: True/False. True means Lya image, False means UV image
    :return: 6 floats and 1 2D array, parameters of the fit. Order: amplitude, 2D array of center_deg [dec, ra],
    center_x, center_y, ellip, sigma_x, sigma_y
    """
    param, sn_r = get_image_params_gaussian2d(image, psf_array, lya, ID, noisy_IDs, high_var_IDs, image_showing, save_image, fitting_method)
    amplitude, center_x, center_y, sigma_x, sigma_y = convert_params_to_variables(param)
    ellip = calc_ellip(sigma_x, sigma_y)
    center_deg = get_image_ra_dec(image, center_x, center_y)

    return amplitude, center_x, center_y, sigma_x, sigma_y, ellip, center_deg, sn_r


def fit_image_smth_else(image, ID, high_var_IDs, noisy_IDs, psf_array, image_showing, save_image,
            fitting_method, lya):
    return 0

def reduce_image(image, cen_x, cen_y, sig_x, sig_y, noisy_IDs, ID, priors = np.zeros(16)):
    min_size = max(51, max(sig_x, sig_y) * 5)
    subimage = (image.subimage((cen_x, cen_y), min_size, unit_center=None, unit_size=None)).copy()
    new_priors = np.copy(priors)
    if ID in noisy_IDs[:, 0]:
        index = np.where(noisy_IDs[:, 0] == ID)[0][0]
        spec_noisy_ID = noisy_IDs[index, :]
        spec_noisy_ID[1] = noisy_IDs[index, 1] - min_size
        spec_noisy_ID[2] = noisy_IDs[index, 2] - min_size
    else:
        spec_noisy_ID = [0, 0, 0, 0, 0]
    for i in range(3, 9):
        if priors[i] != 0:
            new_priors[i] = new_priors[i] - min_size
    return subimage, np.array([spec_noisy_ID, [0, 0, 0, 0, 0]]), new_priors

def clean_image(image, ID, noisy_IDs, high_var_IDs):
    """
    Takes image and if it is marked as noisy image or with high variance, it cleans the image. Otherwise returns
    original data
    :param image: image file
    :param ID: ID of the image
    :param noisy_IDs: IDs of images with high noise
    :param high_var_IDs: 2D array with specific lya/UV IDs of objects with high variance errors
    :return: Returns data 2D array with masked pixels, if they are high variance/noisy (only if mentioned as those IDs)
    """
    data = np.copy(image.data)
    if ID in high_var_IDs[:, 0]:
        index = np.where(high_var_IDs[:, 0] == ID)[0][0]
        data = get_image_with_low_var_errors(image, high_var_IDs[index, :][1])
    if ID in noisy_IDs[:, 0]:
        index = np.where(noisy_IDs[:, 0] == ID)[0][0]
        data = get_clean_image(data, noisy_IDs[index, :])
    return data


def get_image_with_low_var_errors(image, max_std_var):
    """
    Returns data of the image, while removing pixels with high variance. High variance is defined as ones that are
    some amount of standard deviations away from the mean. (Default is 1). That value is defined by the input parameter
    max_std_var
    :param image: Image MPDAF variable
    :param max_std_var: How many st.dev. above the mean should remove the variances
    :return: 2D array data with masked pixels.
    """
    var = image.var
    # image.mask_variance(np.mean(var))
    # max_var = (0.2 * (np.max(var) - np.min(var)) + np.min(var))
    if max_std_var != 0:
        max_var = np.mean(np.nan_to_num(var, nan=0.0)) + max_std_var * np.std(np.nan_to_num(var, nan=0.0))
    else:
        max_var = np.mean(np.nan_to_num(var, nan=0.0)) + default_std_var * np.std(np.nan_to_num(var, nan=0.0))
    data = np.copy(np.where((var > max_var), masking_value, image.data))

    return data

def get_clean_image(data, params):
    """
    Takes in the noisy image and returns the cleaned image
    :param data: The 2D array data of the image
    :param params: parameters of the image. params[i], with i=0: ID, i=1: center_x, i=2: center_y, i=3: tolerance
    from center from which nothing is cleaned (where galaxy is located), i=4: lya or UV
    :return: 2D cleaned data array
    """
    cen_x = params[1]
    cen_y = params[2]
    tol = params[3]

    data_size_x = np.size(data[0])
    total_size = np.size(data)
    data_size_y = int(total_size / data_size_x)

    if cen_x <= 0 or cen_y <= 0:
        cen_x = int(data_size_x / 2)
        cen_y = int(data_size_y / 2)

        tolr = data_size_x * 0.1

        center_array = np.copy(data[int(cen_y - tolr):int(cen_y + tolr), int(cen_x - tolr):int(cen_x + tolr)])

        index_of_cen = np.unravel_index(center_array.argmax(), center_array.shape)
        cen_x = index_of_cen[1] + cen_x - tolr
        cen_y = index_of_cen[0] + cen_y - tolr

        tol = max(int(data_size_x / 4), int(data_size_y / 4))

    tol = min(tol, int(data_size_x / 2 - 1), int(data_size_y / 2 - 1))
    galaxy_center = np.copy(data[max(0, int(cen_y-tol)):min(int(cen_y+tol), data_size_y-1), max(0, int(cen_x-tol)):min(int(cen_x+tol), data_size_x-1)])
    max_val = np.max(np.nan_to_num(galaxy_center, nan=0.0))
    min_val = np.min(np.nan_to_num(galaxy_center, nan=0.0))

    percentage_tol = 0.2

    max_tolerance_value = max((max_val - min_val) * percentage_tol, max_val * percentage_tol) + max(0, min_val)   # Adjust if necessary
    replace_by_value = masking_value      # Adjust if necessary

    new_data = np.copy(data)

    new_data = np.where(new_data>max_tolerance_value, replace_by_value, new_data)

    new_data[max(0, int(cen_y-tol)):min(int(cen_y+tol), data_size_y-1), max(0, int(cen_x-tol)):min(int(cen_x+tol), data_size_x-1)] = galaxy_center
    return new_data


def calc_ellip(sigma_1, sigma_2):
    """
    Calculates ellipticity of the galaxy, given sigma parameters. Order of sigmas doesn't matter
    :param sigma_1: One sigma parameter
    :param sigma_2: Another sigma parameter
    :return: Returns ellipticity of the galaxy
    """
    return 1 - (min(sigma_1, sigma_2) / max(sigma_1, sigma_2))

def calc_ellip_many(sigmas_1, sigmas_2):
    return 1 - (np.min([sigmas_1, sigmas_2], axis=0) / np.max([sigmas_1, sigmas_2], axis=0))


def str_tab(value):  # converts to string and adds tab at the end
    """
    Converts the value to string, replaces dots by commas and adds tab at the end. Used for saving into txt file
    :param value: Value to be converted
    :return: Cleaned string with tab at the end
    """
    return str(value).replace('.', ',') + "\t"

def save_in_txt(text, filename):
    """
    Saves text in file, separating each element in text by tab and adding a new line below it
    :param text: 1D array with words to write
    :param filename: Path and filename where to save
    :return: Returns nothing, but appends the text file
    """
    with open(filename, 'a+') as f:
        for word in text:
            f.write(str_tab(word))
        f.write('\n')

def str_tab_topcat(value):  # converts to string and adds tab at the end
    """
    Converts the value to string, replaces commas by dots and adds tab at the end. Used for saving into txt file.
    Made to be read by TOPCAT (because of dots)
    :param value: Value to be converted
    :return: Cleaned string with tab at the end
    """
    return str(value).replace(',', '.') + "\t"

def save_in_txt_topcat(text, filename):
    """
    Saves text in file, separating each element in text by tab and adding a new line below it. To be read by TOPCAT
    because saves with dots instead of commas.
    :param text: 1D array with words to write
    :param filename: Path and filename where to save
    :return: Returns nothing, but appends the text file
    """
    with open(filename, 'a+') as f:
        for word in text:
            f.write(str_tab_topcat(word))
        f.write('\n')


def get_image_ra_dec(image, x, y):
    """
    Returns the Ra and Dec coordinates of (x,y) pixels in the image
    :param image: The image file
    :param x: x-pixel value
    :param y: y-pixel value
    :return: 1D array [dec, ra] that are dependent on x,y pixels in the image
    """
    return image.wcs.pix2sky([y, x])[0]


def get_image_ra_dec_many(image, x_s, y_s):
    """
    Returns the Ra and Dec coordinates of (x,y) pixels in the image
    :param image: The image file
    :param x: x-pixel value
    :param y: y-pixel value
    :return: 1D array [dec, ra] that are dependent on x,y pixels in the image
    """
    centers = np.array([get_image_ra_dec(image, x_s[0], y_s[0])])
    for i in range(1, np.size(x_s)):
        centers = np.append(centers, [get_image_ra_dec(image, x_s[i], y_s[i])], axis=0)
    return centers


def ang_sep(obj1, obj2):
    """
    Takes [dec, ra] object positions and returns angular separation between the objects in degrees
    :param obj1: 1D array [dec, ra] coordinates of first object
    :param obj2: 1D array [dec, ra] coordinates of second object
    :return: Angular separation between two objects in degrees
    """
    ra1 = obj1[1]
    dec1 = obj1[0]
    ra2 = obj2[1]
    dec2 = obj2[0]
    return np.arccos(np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))

def ang_sep_many(arr1, arr2):
    """
    Takes [dec, ra] object positions and returns angular separation between the objects in degrees
    :param arr1: 2D array, each element in format [dec, ra] coordinates of first object
    :param arr2: 2D array, each element in format [dec, ra] coordinates of second object
    :return: 1D array Angular separation between two objects in degrees
    """
    ra1 = arr1[:, 1]
    dec1 = arr1[:, 0]
    ra2 = arr2[:, 1]
    dec2 = arr2[:, 0]
    return np.arccos(np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))

def kpc_sep(ang_sep, z):
    """
    Given angular separation in arcseconds and the redshift z, what is the angular separation between two objects.
    Assumes flat model with h0 = 0.7, baryon mass = 0.3.
    :param ang_sep: Angular separation between two objects in arcseconds
    :param z: Redshift to objects z
    :return: Distance between two objects in kpc
    """
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    d_A = cosmo.angular_diameter_distance(z=z)
    theta = ang_sep / 3600 # convert from arcsec to degrees
    theta_radian = theta * np.pi / 180
    distance_kpc = d_A * theta_radian
    return distance_kpc / u.Mpc * u.kpc * 1000

def kpc_sep_many(ang_sep_arr, z):
    """
    Given angular separation in arcseconds and the redshift z, what is the angular separation between two objects.
    Assumes flat model with h0 = 0.7, baryon mass = 0.3. Can inside arrays as well
    :param ang_sep: Angular separation between two objects in arcseconds
    :param z: Redshift to objects z
    :return: Distance between two objects in kpc
    """
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    d_A = cosmo.angular_diameter_distance(z=z)
    theta = ang_sep_arr / 3600  # convert from arcsec to degrees
    theta_radian = theta * np.pi / 180
    distance_kpc = d_A * theta_radian
    return distance_kpc / u.Mpc * u.kpc * 1000


def get_noisy_IDs(path_textfile, default_cenx = 0, default_ceny = 0, default_tol = 18):
    """
    Takes in path textfile to the file that contains IDs with lots of noise of bad pixels. It then returns 2D array
    with the values.
    :param path_textfile: Path to the textfile in format:
    ID:int (ID with noisy image)
    center_x:int (the theoretical center in pixels of galaxy in x-axis). 0 will convert to default value
    center_y:int (the theoretical center in pixels of galaxy in y-axis). 0 will convert to default value
    tolerance:int (how big is the galaxy in pixels; i.e. ignore any abnormalities from center within tolerance).
    0 will convert to default value
    lya:0 or 1. 0 means UV, 1 means Lya.
    :return: 2D array with the bad pixel values. values[i] gives the speicifc rows.
    values[i][j] gives specific element. j=0: ID, j=1: center_x, j=2: center_y, j=3: tolerance, j=4: 0 or 1 depending
    if it is Lya (then 1) or UV (then 0)
    """

    values = np.loadtxt(path_textfile).astype(int)
    values[:, 1][values[:, 1] == 0] = default_cenx
    values[:, 2][values[:, 2] == 0] = default_ceny
    values[:, 3][values[:, 3] == 0] = default_tol
    return values

def load_file_as_integers(path_textfile):
    """
    Takes in path textfile to the file. It then returns 2D array with the values from file. All values returned are integers.
    :param path_textfile: Path to the textfile in format
    :return: 2D array with the integer values. values[i] gives the speicifc rows.
    values[i][j] gives specific element
    """

    values = np.loadtxt(path_textfile).astype(int)
    return values


def get_ids_and_z(values, psf_values, IDs_to_find, start_at_ID=0):
    """
    Takes 2D array with values and 1D array with IDs to use and returns seven 1D arrays. Finds the required IDs
    given values and returns the arrays with corresponding values
    :param values: Values with all stats in the form of "ID, RID, Ra, Dec, z, Ra, Dec". First ra/dec are Lya,
    second ones are UV.
    :param IDs_to_find: Which IDs to find among the values
    :return: Returns 7 1D arrays in the same order as values ("ID, RID, Ra, Dec, z, Ra, Dec"). Those represent
    ID with corresponding values.
    """
    index_to_start = 0
    if start_at_ID > 0:
        index_to_start = np.where(values[:,0].astype(int) == start_at_ID)[0]
        if index_to_start:
            index_to_start = index_to_start[0]
        else:
            index_to_start = 0

    values = values[values[:, 0].argsort()]
    psf_values = psf_values[psf_values[:, 0].argsort()]

    IDs = values[index_to_start:, 0].astype(int)
    Rid = values[index_to_start:, 1].astype(int)
    Ra_Lya = values[index_to_start:, 2]
    Dec_Lya = values[index_to_start:, 3]
    z_s = values[index_to_start:, 4]
    Ra_UV = values[index_to_start:, 5]
    Dec_UV = values[index_to_start:, 6]

    IDs_psf = psf_values[index_to_start:, 0].astype(int)
    alpha_psf = psf_values[index_to_start:, 1]
    beta_psf = psf_values[index_to_start:, 2]
    sigma_psf = psf_values[index_to_start:, 3]

    IDs_to_use = np.array([])
    RIDs_to_use = np.array([])
    Ra_Lya_to_use = np.array([])
    Dec_Lya_to_use = np.array([])
    z_s_to_use = np.array([])
    Ra_UV_to_use = np.array([])
    Dec_UV_to_use = np.array([])

    IDs_psf_to_use = np.array([])
    alpha_psf_to_use = np.array([])
    beta_psf_to_use = np.array([])
    sigma_psf_to_use = np.array([])

    for i in range(np.size(IDs_to_find)):
        index_to_use = np.where(IDs == IDs_to_find[i])[0]
        if len(index_to_use) != 0:
            j = index_to_use[0]
            IDs_to_use = np.append(IDs_to_use, int(str(int(IDs[j])).replace(',','').replace('.',''))).astype(int)
            RIDs_to_use = np.append(RIDs_to_use, int(str(int(Rid[j])).replace(',','').replace('.',''))).astype(int)
            Ra_Lya_to_use = np.append(Ra_Lya_to_use, Ra_Lya[j])
            Dec_Lya_to_use = np.append(Dec_Lya_to_use, Dec_Lya[j])
            z_s_to_use = np.append(z_s_to_use, z_s[j])
            Ra_UV_to_use = np.append(Ra_UV_to_use, Ra_UV[j])
            Dec_UV_to_use = np.append(Dec_UV_to_use, Dec_UV[j])

            if int(IDs[j]) == int(IDs_psf[j]):
                IDs_psf_to_use = np.append(IDs_psf_to_use, int(str(int(IDs_psf[j])).replace(',','').replace('.',''))).astype(int)
                alpha_psf_to_use = np.append(alpha_psf_to_use, alpha_psf[j])
                beta_psf_to_use = np.append(beta_psf_to_use, beta_psf[j])
                sigma_psf_to_use = np.append(sigma_psf_to_use, sigma_psf[j])
            else:
                SystemExit("Error, psf and ID values are NOT consistent with each other")

    return IDs_to_use, RIDs_to_use, Ra_Lya_to_use, Dec_Lya_to_use, z_s_to_use, Ra_UV_to_use, Dec_UV_to_use, \
           IDs_psf_to_use, alpha_psf_to_use, beta_psf_to_use, sigma_psf_to_use

def generate_IDs_to_analyse(image_analysis_filepath):
    """
    Opens the file and analysis which ones are isolated images with and without HST.
    :param image_analysis_filepath: path to the analysis text file with IDs and stuff. The filename is expected
    to contain five columns. First is integer, 4 others are 1 or 0 (True or False):
    "ID, Isolated, Centered, 3rd panel OK, HST exists"
    :return: Returns nothing directly, but creates two text files: "isolated_HST_IDs.txt" and
    "isolated_no_HST_IDs.txt" with appropriate IDs.
    """
    values = np.loadtxt(image_analysis_filepath).astype(int)
    isolated_IDs = []
    hst_exists_IDs = []

    for i in range(int(np.size(values) / np.size(values[0]))):
        if values[i][1] == 1 and values[i][4] == 0:   # isolated and no HST
            isolated_IDs.append(values[i][0])
        if values[i][1] == 1 and values[i][4] == 1:   # HST exists and isolated
            hst_exists_IDs.append(values[i][0])

    with open(filepath_HST_IDs, 'w+') as f:
        for word in hst_exists_IDs:
            f.write(str(word) + '\n')

    with open(filepath_no_HST_IDs, 'w+') as f:
        for word in isolated_IDs:
            f.write(str(word) + '\n')

def get_HST_IDs():
    return np.loadtxt(filepath_HST_IDs).astype(int)

def get_no_HST_IDs():
    return np.loadtxt(filepath_no_HST_IDs).astype(int)

def get_uncertainties(ID, image, psf_array, number_of_slices, noisy_IDs, high_var_IDs, model, fitting_method, lya, priors=np.zeros(16)):
    """
    Gets UV, Lya image and creates number_of_slices images and calculates uncertainties
    :param ID: ID that is used (for naming purposes only)
    :param image: Image data of lya
    :param image_uv: Image data of UV
    :param psf_array: 1D array with PSF parameters. psf_array[j] gives 1D array of specific values. j=0: ID, j=1: alpha, j=2: beta,
    j=3: sigma of PSF.
    :param number_of_slices: How many images to test (1000 should be standard, but slow)
    :return: Returns 1D array with uncertainties in the following order:
    [amp_lya, cen_x_lya, cen_y_lya, sig_x_lya, sig_y_lya, ellip_lya, amp_uv, cen_x_uv, cen_y_uv, sig_x_uv, sig_y_uv,
    ellip_uv, ang_separation, kpc_separation]
    """
    amplitudes, center_x_s, center_y_s, cube_boot, ellipticity, error, sigma_x_s, sigma_y_s, x, y = prepare_uncert_calcs(
        image, number_of_slices, ID, noisy_IDs, high_var_IDs, lya)

    if lya:
        name = "Lya"
    else:
        name = "UV"

    if number_of_slices >= threshold_for_workers and not debug_mode:
        amplitudes = np.array([])
        center_x_s = np.array([])
        center_y_s = np.array([])
        ellipticity = np.array([])
        sigma_x_s = np.array([])
        sigma_y_s = np.array([])
        center_deg = np.array([[]])
        futures = []
    else:
        center_deg = np.zeros((number_of_slices, 2))

    if model == "gaussian2D":
        if number_of_slices >= threshold_for_workers and not debug_mode:
            print("starting uncertainty calculations")
            for im_boot in cube_boot:
                future = client.submit(gauss2d_fitting_no_plot_params, im_boot.data, x, y, error, psf_array, lya, fitting_method, priors)
                futures.append(future)

            print("start gathering")
            futures = np.array(client.gather(futures))
            amplitudes = futures[:, 0, 0]
            center_x_s = futures[:, 0, 1]
            center_y_s = futures[:, 0, 2]
            sigma_x_s = futures[:, 0, 3]
            sigma_y_s = futures[:, 0, 4]
            ellipticity = calc_ellip_many(sigma_x_s, sigma_y_s)
            center_deg = get_image_ra_dec_many(image, center_x_s, center_y_s)

            #center_x_s = np.append(center_x_s, futures['centerx'] * 1)
            #center_y_s = np.append(center_y_s, futures['centery'] * 1)
            #sigma_x_s = np.append(sigma_x_s, futures['sigmax'] * 1)
            #sigma_y_s = np.append(sigma_y_s, futures['sigmay'] * 1)
            #ellipticity = np.append(ellipticity, calc_ellip(futures['sigmax'] * 1, futures['sigmay'] * 1))
            #center_deg = np.append(center_deg, [get_image_ra_dec(image, futures['centerx'] * 1, futures['centery'] * 1)], axis=1)


            #amplitudes = client.gather(amplitudes)
            #amplitudes = client.gather(amplitudes)
            #center_x_s = client.gather(center_x_s)
            #center_y_s = client.gather(center_y_s)
            #sigma_x_s = client.gather(sigma_x_s)
            #sigma_y_s = client.gather(sigma_y_s)
            #ellipticity = client.gather(ellipticity)
            #center_deg = client.gather(center_deg)
            print("Worker calculation done")
        else:
            for j in range(0, number_of_slices):
                im_boot = cube_boot[j]
                parameters = gauss2d_fitting_no_plot_params_img(im_boot.data, x, y, image.var, psf_array, lya, debug_mode, fitting_method, priors=priors)

                amplitudes[j], center_x_s[j], center_y_s[j], sigma_x_s[j], sigma_y_s[j] = convert_params_to_variables(parameters)
                ellipticity[j] = calc_ellip(sigma_x_s[j], sigma_y_s[j])
                center_deg[j] = get_image_ra_dec(image, center_x_s[j], center_y_s[j])

                if j % 20 == 0:
                   print(int(j / number_of_slices * 100), "% done for image", name, ID)

    return [np.std(amplitudes), np.std(center_x_s), np.std(center_y_s), np.std(sigma_x_s),
            np.std(sigma_y_s), np.std(ellipticity), center_deg]


def prepare_uncert_calcs(image, number_of_slices, ID, noisy_IDs, high_var_IDs, lya):
    """
    Prepares the number_of_slices cubes for the uncertainty calculations
    :param image: image MPDAF file
    :param number_of_slices: number of slices for uncertainty calculations
    :param ID: ID of object
    :param noisy_IDs: IDs of noisy objects
    :param high_var_IDs: 2D array with IDs of objects with high variance errors
    :param lya: True for lya, False for UV
    :return: Returns 10 lists: empty amplitude, center_x, center_y, cube_boot (full of cubes randomly changed within
    variance for uncertainty calculations), empty ellipticity, flattened error, empty sigma_x, sigma_y, flattened x, y
    """
    imlist = []
    if lya:
        data = clean_image(image, ID, noisy_IDs[noisy_IDs[:, 4] == 1], high_var_IDs[high_var_IDs[:, 2] == 1])
    else:
        data = clean_image(image, ID, noisy_IDs[noisy_IDs[:, 4] == 0], high_var_IDs[high_var_IDs[:, 2] == 0])
    for d, v in zip(data.ravel(), image.var.ravel()):
        imlist.append(np.random.normal(d, np.sqrt(v), size=number_of_slices))
    tab_l = np.vstack(imlist).T
    tab_l = tab_l.reshape((number_of_slices,) + image.shape)
    cube_boot = Cube(data=tab_l)
    amplitudes = np.zeros(number_of_slices)
    center_x_s = np.zeros(number_of_slices)
    center_y_s = np.zeros(number_of_slices)
    sigma_x_s = np.zeros(number_of_slices)
    sigma_y_s = np.zeros(number_of_slices)
    ellipticity = np.zeros(number_of_slices)
    x, y, error = flatten_xy_err(cube_boot[0].data, image.var)
    return amplitudes, center_x_s, center_y_s, cube_boot, ellipticity, error, sigma_x_s, sigma_y_s, x, y





def get_image_params_gaussian2d(image, psf_array, lya, ID, noisy_IDs, high_var_IDs, image_showing, save_image, fitting_method):
    """
    Takes image and variables and returns the parameters of the Gaussian2D fit
    :param image: MPDAF image fit
    :param psf_array: 1D array with PSF parameters. psf_array[j] gives values for PSF for that ID. j=0: ID, j=1: alpha,
    j=2: beta, j=3: sigma of PSF
    :param lya: True/False. True means Lya image, False means UV image
    :param ID: int ID of the image
    :param noisy_IDs: 2D array with IDs that need to be masked. Format: ID, center_x (in pixels), center_y
    (in pixels), tolerance (in pixels, within tolerance no masking), lya (1 is lya, 0 is uv)
    :param high_var_IDs: 2D array with IDs that need high variance pixels to be masked. Format: ID,
    variance tolerance (float; variance with var_mean + var_tol * var_st.dev are masked), lya (1 is lya, 0 is uv)
    :param image_showing: True/False whether to show image
    :param save_image: True/False whether to save image into an output folder
    :param fitting_method: 2D array with fitting methods. Format: ID, integer for method (e.g. 0 is
    leastsq, 1 is least_squares), lya (1 is lya, 0 is uv)
    :return: Parameters tuple with parameters of the Gaussian2D fit.
    """

    if lya:
        data = clean_image(image, ID, noisy_IDs[noisy_IDs[:, 4] == 1], high_var_IDs[high_var_IDs[:, 2] == 1])
    else:
        data = clean_image(image, ID, noisy_IDs[noisy_IDs[:, 4] == 0], high_var_IDs[high_var_IDs[:, 2] == 0])
    x, y, error = flatten_xy_err(data, image.var)

    method = get_fitting_method(ID, fitting_method)

    out, mod = gauss2d_fitting_no_plot(data, x, y, error, psf_array, lya, fitting_method=method)

    if image_showing or save_image:
        if lya:
            name = "Lya_of_" + str(ID)
        else:
            name = "UV_of_" + str(ID)
        plot_image(out, data, x, y, image.var, mod, name, image_showing, save_image)
    params = out.params
    sn_r = get_sn(data, out, mod, x, y)
    return params, sn_r


def get_sn(og_data, out, mod, x, y):
    x_max = int(np.max(x) + 1)
    y_max = int(np.max(y) + 1)
    X, Y = np.meshgrid(np.linspace(np.min(x), np.max(y), x_max),  # Converts x,y,z values to meshgrid for drawing
                       np.linspace(np.min(y), np.max(y), y_max))
    fit = mod.func(X, Y, **out.best_values)
    error = np.mean(np.square(np.nan_to_num(convert_data_to_odd_axes(og_data), nan=0.0) - fit))
    signal_peak = np.mean(np.square(fit))
    sn = signal_peak / error
    return sn


def get_sn_scipy_old(img, axis=None, ddof=0):
    """
    The signal-to-noise ratio of the input data.
    Returns the signal-to-noise ratio of `a`, here defined as the mean
    divided by the standard deviation.
    Parameters
    ----------
    a : array_like
        An array_like object containing the sample data.
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over
        the whole array `a`.
    ddof : int, optional
        Degrees of freedom correction for standard deviation. Default is 0.
    Returns
    -------
    s2n : ndarray
        The mean to standard deviation ratio(s) along `axis`, or 0 where the
        standard deviation is 0.
    """
    a = np.asanyarray(img)
    m = a.mean(axis=axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)