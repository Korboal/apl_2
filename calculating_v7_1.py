import numpy as np
import astropy.units as u
from mpdaf.obj import Image, Cube
from gauss_fitting_v7_1 import gauss2d_fitting_no_plot, flatten_xy_err, plot_image, gauss2d_fitting_no_plot_params, \
    convert_params_to_variables, gauss2d_fitting_no_plot_params_img, convert_data_to_odd_axes
from astropy.cosmology import FlatLambdaCDM
from dask.distributed import Client
import matplotlib

filepath_HST_IDs = "isolated_HST_IDs.txt"       # path to the HST IDs
filepath_no_HST_IDs = "isolated_no_HST_IDs.txt"     # path to the IDs only Lya (no UV)
masking_value = np.nan      # Masking value. nan should work best, although it is basically working like 0
default_std_var = 3         # for high variance masking. if variance > default_std_Var * mean_variance, then it is masked
threshold_for_workers = 20      # minimum amount of slices required to use multiprocessing (set to >1000 to basically never have multiprocessing)
workers = 12  # should be the same as cores
threads_per_worker = 1  # seemed to work best with 1; play around if you want.
debug_mode = False  # debug mode to save pictures while uncertainty calc
do_resampling = True    # Whether do resampling
auto_detect_resample = True     # for now same behaviour as do_resampling. need both True for resampling
def_resample_coeff = 2  # starting resampling coefficient
max_resample_coff = 4   # maximum resampling coefficient. beyond 4 might take very long time
min_sigma_for_resample = 1.0    # sigma required to stop resampling. if sigma < this variable, then increase resampling

def new_fitting_with_image(paths_Lya, paths_UV, psf_array, IDs, z_s, Ra_lya, Dec_lya, Ra_uv, Dec_uv, filename_val_only, number_of_slices, lya_fit, uv_fit,
                           uncertainty_calc,  image_info, image_showing, save_image,
                           noisy_IDs=np.array([[0]]), high_var_IDs=np.array([[0]]), model_lya=np.array([[0]]), model_uv=np.array([[0]]),
                           fitting_method_lya=np.array([[0]]), fitting_method_uv=np.array([[0]])):
    """
    Calculates and fits values and uncertainty for center, sigma, ellipticity and separation. One function that does everything
    :param paths_Lya: 1D array with the paths to the Lya images
    :param paths_UV: 1D array with the path to the UV images
    :param psf_array: 2D array with PSF parameters. psf_array[j] gives 1D array of specific values. j=0: ID, j=1: alpha, j=2: beta,
    j=3: sigma of PSF. psf_array[j][i] gives i-th 1D array with j-th parameter for the specific image. alpha, beta are
    MOFFAT2D Kernel parameters for PSF (repsectively called gamma, alpha in astropy.convolution). sigma is Gaussian2D
    kernel paramter
    :param IDs: 1D array the IDs to use
    :param z_s: 1D array the redshift z of the images
    :param Ra_lya: 1D array the given Ra Lya value (where the image was probed; does not do anything in the program,
    only for saving in the text file; can be just zeroes in theory)
    :param Dec_lya: 1D array the given Dec Lya value (where the image was probed; does not do anything in the program,
    only for saving in the text file; can be just zeroes in theory)
    :param Ra_uv: 1D array the given Ra UV value (where the image was probed; does not do anything in the program,
    only for saving in the text file; can be just zeroes in theory)
    :param Dec_uv: 1D array the given Dec UV value (where the image was probed; does not do anything in the program,
    only for saving in the text file; can be just zeroes in theory)
    :param filename_val_only: String - Path + filename where to save values
    :param number_of_slices: integer - number of slices to do uncertainty calculations (1000 usually standard, less for
    debugging)
    :param lya_fit: True/False if need to do Lya fit
    :param uv_fit: True/False if need to do UV fit
    :param uncertainty_calc: True/False whether to do uncertainty calculations using number_of_slices
    :param image_info: True/False whether to show image information
    :param image_showing: True/False whether to show fitted image (does not work when doing multithreaded fitting)
    :param save_image: True/False whether to save image into an output folder
    :param noisy_IDs: optional, int 2D array with IDs that need to be masked. Format: ID, center_x (in pixels), center_y
    (in pixels), tolerance (in pixels, within tolerance no masking), lya (1 is lya, 0 is uv)
    :param high_var_IDs: optional, int 2D array with IDs that need high variance pixels to be masked. Format: ID,
    variance tolerance (int; variance with var_mean + var_tol * var_st.dev are masked), lya (1 is lya, 0 is uv)
    :param model_lya: optional, 2D array with models. Format: ID (int), model (int; 0 = gaussian2D). Append here for
    mode models  TODO add more models
    :param model_uv: optional, 2D array with models. Format: ID (int), model (int; 0 = gaussian2D). Append here for
    mode models  TODO add more models
    :param fitting_method_lya: optional, 2D array with fitting methods. Format: ID, integer for method (e.g. 0 is
    leastsq, 1 is least_squares), lya (1 is lya, 0 is uv). Default assumed is least_squares
    :param fitting_method_uv: optional, 2D array with fitting methods. Format: ID, integer for method (e.g. 0 is
    leastsq, 1 is least_squares), lya (1 is lya, 0 is uv). Default assumed is least_squares
    :return: Returns nothing, but appends filenames with values. Text saved in following order (if only Lya, skips
    UV and separation parts; saved for use in TOPCAT program, just load table as ASCII; can probably copy to excel
    as well):
    #ID z amp_l d_amp_l cen_x_l d_cen_x_l cen_y_l d_cen_y_l sig_x_l d_sig_x_l sig_y_l d_sig_y_l ell_l d_ell_l
    amp_u d_amp_u cen_x_u d_cen_x_u cen_y_u d_cen_y_u sig_x_u d_sig_x_u sig_y_u d_sig_y_u ell_u d_ell_u
    ang_sep d_ang_sep kpc_sep d_kpc_sep lya_ra_fit lya_dec_fit lya_ra lya_dec uv_ra_fit uv_dec_fit uv_ra uv_dec
    ID - ID of the object.
    z - redshift z of the object
    amp - Amplitude
    cen_x - center position in pixel value along x-axis
    cen_y - center position in pixel value along y-axis
    sig_x - sigma of Gaussian fit along x-axis
    sig_y - sigma of Gaussian fit along y-axis
    ell - ellipticity
    ang_sep - separation/offset of UV and Lya centers in arcsec
    kpc_sep - separation/offset of UV and Lya centers in kpc
    _ra_fit/_dec_fit - center in ra/dec coordinates of the fit
    _ra/_dec - center in ra_dec coordinates of the original image (compare to fit; should be close, but not exact)
    SOMETHING_l - SOMETHING of Lya fit
    SOMETHING_u - SOMETHING of UV fit
    d_SOMETHING - uncertainty in SOMETHING
    """
    resample_coeff = def_resample_coeff     # If doing resampling, the initial resampling coefficient starts at default value

    if uncertainty_calc and number_of_slices >= threshold_for_workers and not debug_mode:   # For multiprocessing. Remove if multiprocessing not needed
        matplotlib.use('Agg')   # Does not work without this (no idea why). Disables image saving??
        print("Preparing workers")
        global client       # Global variable, because passing along functions is annoying
        client = Client(threads_per_worker=threads_per_worker, n_workers=workers)   # our client for actual multiprocessing
        print("Worker preparation complete")

    if uncertainty_calc and lya_fit and uv_fit:     # prepares text file to save values/uncertainties
        save_in_txt(["#ID\tz\tamp_l\td_amp_l\tcen_x_l\td_cen_x_l\tcen_y_l\td_cen_y_l\tsig_x_l\td_sig_x_l\tsig_y_l\td_sig_y_l\t"
                        "ell_l\td_ell_l"
                        "\tamp_u\td_amp_u\tcen_x_u\td_cen_x_u\tcen_y_u\td_cen_y_u\tsig_x_u\td_sig_x_u\tsig_y_u\td_sig_y_u\t"
                        "ell_u\td_ell_u"
                        "\tang_sep\td_ang_sep\tkpc_sep\td_kpc_sep\tlya_ra_fit\tlya_dec_fit\tlya_ra\tlya_dec\tuv_ra_fit\t"
                        "uv_dec_fit\tuv_ra\tuv_dec\tsn_l\tsn_u"], filename_val_only)

    if uncertainty_calc and lya_fit and not uv_fit: # prepares text file to save values/uncertainties, if no HST/UV
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

            max_resample_reached = False    # By default, resets the variable

            if auto_detect_resample and do_resampling:  # If resampling is done, then actually do it
                do_resample = False     # initially assume resampling not needed
                amplitude_lya, center_deg_lya, center_x_lya, center_y_lya, ellip_lya, sigma_x_lya, sigma_y_lya, s_n_r_lya = fit_image(
                    IDs[i], fitting_method_lya, high_var_IDs, image_lya, image_showing, model_lya, noisy_IDs,
                    psf_array[:, i],
                    save_image, True)       # do first fit, see the values
                prior_arr_lya = get_prior_array(amplitude_lya, center_x_lya, center_y_lya, sigma_x_lya, sigma_y_lya)
                if sigma_x_lya < min_sigma_for_resample or sigma_y_lya < min_sigma_for_resample:    # if sigma too low, do resampling
                    do_resample = True
                    for j in range(def_resample_coeff, max_resample_coff+1):    # fits and increase resampling until reaching max value
                        print("Resampling with coefficient", resample_coeff)
                        amplitude_lya, center_deg_lya, center_x_lya, center_y_lya, ellip_lya, sigma_x_lya, sigma_y_lya, s_n_r_lya = fit_image(
                            IDs[i], fitting_method_lya, high_var_IDs, image_lya, image_showing, model_lya, noisy_IDs, psf_array[:, i],
                            save_image, True, do_resample=do_resample, resample_coeff=resample_coeff)
                        prior_arr_lya = get_prior_array(amplitude_lya, center_x_lya, center_y_lya, sigma_x_lya,
                                                        sigma_y_lya)
                        center_x_lya, center_y_lya, sigma_x_lya, sigma_y_lya = center_x_lya/resample_coeff, center_y_lya/resample_coeff, sigma_x_lya/resample_coeff, sigma_y_lya/resample_coeff
                        if sigma_x_lya*resample_coeff >= min_sigma_for_resample and sigma_y_lya*resample_coeff >= min_sigma_for_resample:   # if sigma is good enough now, stop resampling
                            print("Resampling finished at", resample_coeff)
                            break
                        else:
                            if resample_coeff < max_resample_coff:  # increases resampling values
                                resample_coeff += 1
                            else:       # max resampling value reached, stop fitting
                                max_resample_reached = True     # makes fittings ignore this image, since more resampling than threshold is needed
                                print("Max resampling coefficient of", resample_coeff, "reached")
            else:       # if resampling is assumed to be never needed
                do_resample = False
                amplitude_lya, center_deg_lya, center_x_lya, center_y_lya, ellip_lya, sigma_x_lya, sigma_y_lya, s_n_r_lya = fit_image(
                    IDs[i], fitting_method_lya, high_var_IDs, image_lya, image_showing, model_lya, noisy_IDs,
                    psf_array[:, i],
                    save_image, True)   # just do fitting normally
                prior_arr_lya = get_prior_array(amplitude_lya, center_x_lya, center_y_lya, sigma_x_lya, sigma_y_lya)    # priors for uncertainty calculations

            if uncertainty_calc and not max_resample_reached:    # uncertainty calculations for Lya fit
                spec_model_lya = get_model_name(IDs[i], model_lya)  # Checks which model to fit for our image. By default Gaussian. Add more models here and in uncertainty calculations if you add more models
                spec_fitting_method_lya = get_fitting_method(IDs[i], fitting_method_lya)    # returns name of the fitting method, if not default

                uncertainties_lya = get_uncertainties(IDs[i], image_lya, psf_array[:, i], number_of_slices,
                                                      noisy_IDs, high_var_IDs, spec_model_lya, spec_fitting_method_lya,
                                                      True, priors=prior_arr_lya, do_resample=do_resample,
                                                      resample_coeff=resample_coeff) # Uncertainty calculations
                d_amp_l, d_cen_x_l, d_cen_y_l, d_sig_x_l, d_sig_y_l, d_ellip_l, centers_deg_l = convert_uncertainties_to_variables(
                    uncertainties_lya)  # saves parameters as separate values (most of them already are float st.dev., except centers_deg

                if not uv_fit:  # saves uncertainties + values if no HST; to skip separation/offset values
                    print("Full calculation done for Lya", IDs[i])
                    save_in_txt_topcat([IDs[i], z_s[i], amplitude_lya, d_amp_l, center_x_lya, d_cen_x_l,
                                        center_y_lya, d_cen_y_l, sigma_x_lya, d_sig_x_l, sigma_y_lya,
                                        d_sig_y_l,
                                        ellip_lya, d_ellip_l,
                                        center_deg_lya[1], center_deg_lya[0], Ra_lya[i], Dec_lya[i], s_n_r_lya],
                                       filename_val_only)
            resample_coeff = def_resample_coeff # reset the variables for refitting
            do_resample = False

        if uv_fit:  # UV fit; same as Lya mostly
            image_uv = Image(paths_UV[i])
            print("UV image of", IDs[i])
            if image_info:
                print(image_uv.info())

            if auto_detect_resample and do_resampling:
                do_resample = False
                amplitude_uv, center_deg_uv, center_x_uv, center_y_uv, ellip_uv, sigma_x_uv, sigma_y_uv, s_n_r_uv = fit_image(
                    IDs[i], fitting_method_uv, high_var_IDs, image_uv, image_showing, model_uv, noisy_IDs,
                    psf_array[:, i],
                    save_image, False)
                prior_arr_uv = get_prior_array(amplitude_uv, center_x_uv, center_y_uv, sigma_x_uv, sigma_y_uv)
                if sigma_x_uv < min_sigma_for_resample or sigma_y_uv < min_sigma_for_resample:
                    do_resample = True
                    for j in range(def_resample_coeff, max_resample_coff+1):
                        print("Resampling with coefficient", resample_coeff)
                        amplitude_uv, center_deg_uv, center_x_uv, center_y_uv, ellip_uv, sigma_x_uv, sigma_y_uv, s_n_r_uv = fit_image(
                            IDs[i], fitting_method_uv, high_var_IDs, image_uv, image_showing, model_uv, noisy_IDs,
                            psf_array[:, i],
                            save_image, False, do_resample=do_resample, resample_coeff=resample_coeff)
                        prior_arr_uv = get_prior_array(amplitude_uv, center_x_uv, center_y_uv, sigma_x_uv, sigma_y_uv)
                        center_x_uv, center_y_uv, sigma_x_uv, sigma_y_uv = center_x_uv / resample_coeff, center_y_uv / resample_coeff, sigma_x_uv / resample_coeff, sigma_y_uv / resample_coeff
                        if sigma_x_uv*resample_coeff >= min_sigma_for_resample and sigma_y_uv*resample_coeff >= min_sigma_for_resample:
                            print("Resampling finished at", resample_coeff)
                            break
                        else:
                            if resample_coeff < max_resample_coff:
                                resample_coeff += 1
                            else:
                                print("Max resampling coefficient of", resample_coeff, "reached")
            else:
                do_resample = False
                amplitude_uv, center_deg_uv, center_x_uv, center_y_uv, ellip_uv, sigma_x_uv, sigma_y_uv, s_n_r_uv = fit_image(
                    IDs[i], fitting_method_uv, high_var_IDs, image_uv, image_showing, model_uv, noisy_IDs,
                    psf_array[:, i],
                    save_image, False)
                prior_arr_uv = get_prior_array(amplitude_uv, center_x_uv, center_y_uv, sigma_x_uv, sigma_y_uv)

            if uncertainty_calc:    # uncertainty calculations for UV
                spec_model_uv = get_model_name(IDs[i], model_uv)
                spec_fitting_method_uv = get_fitting_method(IDs[i], fitting_method_uv)

                image_uv_2, spec_noisy_ID, prior_arr_uv = reduce_image(image_uv, center_x_uv, center_y_uv, sigma_x_uv, sigma_y_uv,
                                                     noisy_IDs[noisy_IDs[:, 4] == 0], IDs[i], priors=prior_arr_uv,
                                                                       do_resample=do_resample, resample_coeff=resample_coeff)
                # reduce image size. HST has usually been several times bigger than Lya, leading to VERY slow (up to
                # several hours long) uncertainty calculations. Here, it crops the image around the center of galaxy
                # (which is known from first fit), down to size of galaxy or minimum image size (written in function).
                # Then it fixes values to pass for the uncertainty calculations

                uncertainties_uv = get_uncertainties(IDs[i], image_uv_2, psf_array[:, i], number_of_slices, spec_noisy_ID,
                                                     high_var_IDs, spec_model_uv, spec_fitting_method_uv, False, priors=prior_arr_uv, do_resample=do_resample, resample_coeff=resample_coeff)
                d_amp_u, d_cen_x_u, d_cen_y_u, d_sig_x_u, d_sig_y_u, d_ellip_u, centers_deg_u = convert_uncertainties_to_variables(
                    uncertainties_uv)

            resample_coeff = def_resample_coeff
            do_resample = False

        if uncertainty_calc and lya_fit and uv_fit:  # lya and hst counter-part; calculates separations
                lya_uv_ang_sep = ang_sep(center_deg_lya, center_deg_uv) * 3600  # converts degrees to arcsec
                diff_in_kpc = kpc_sep(lya_uv_ang_sep, z_s[i]) / u.kpc   # remove units

                lya_uv_ang_seps = ang_sep_many(centers_deg_l, centers_deg_u) * 3600  # converts degrees to arcsec
                diff_in_kpcs = kpc_sep_many(lya_uv_ang_seps, z_s[i]) / u.kpc    # remove units

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
    """
    For uncertainty calculations, we can know exact center and sigma values. Here, I create an array with priors to pass
    as starting values + limits for the fitting.
    :param amplitude: Float fit amplitude
    :param center_x: Float center_x in pixels
    :param center_y: Float center_y in pixels
    :param sigma_x: Float sigma_x in pixels
    :param sigma_y: Float sigma_y in pixels
    :return: 1D array with 16 items. They act as priors+limits for the fitting. 0 indicates to use default value in
    fitting. The 1D array is made in following order: [amplitude, lower_limit_for_amp, upper_lim_for_amp, center_x,
    low_limit_cen_x, upper_lim_cen_x, center_y, low_limit_cen_y, upper_lim_cen_y, sigma_x, low_limit_sig_x,
    upper_lim_sig_x, sigma_y, low_limit_sig_y, upper_lim_sig_y, initial_rotation]. IF initial rotation is included,
    make sure it is within the boundaries of the fitting.
    """
    cen_sig_tol_offset = 5  # limits for center_x/y. The limits will be given as center +/- sigma * cen_sig_tol_offset. I.e. within how many sigmas, do we limit our center
    sig_tol = 2  # limits for sigma. Limits are sigma * sig_tol and sigma / sig_tol. I.e. multiplied/divided by the sig_tol
    prior_arr = np.array(
        [amplitude, 0, 0, center_x, center_x - cen_sig_tol_offset * sigma_x, center_x + cen_sig_tol_offset * sigma_x, center_y,
         center_y - cen_sig_tol_offset * sigma_y, center_y + cen_sig_tol_offset * sigma_y, sigma_x, sigma_x / sig_tol, sigma_x * sig_tol,
         sigma_y, sigma_y / sig_tol, sigma_y * sig_tol, 0])
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

def fit_image(ID, fitting_method, high_var_IDs, image, image_showing, model, noisy_IDs, psf_array, save_image, lya, do_resample=False, resample_coeff=1):
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
    :param do_resample: optional, True/False. False by default. If true, it will do resampling
    :param resample_coeff: optional, integer. 1 by defaultt. If do_resample True, then will do resample by the
    coefficient amount
    :return: 7 floats and 1 2D array, parameters of the fit. Order: amplitude, 2D array of center_deg [dec, ra],
    center_x, center_y, ellip, sigma_x, sigma_y, sn_r
    """
    fit_model = get_model_name(ID, model)   # by default returns 'gaussian2D'; i.e. convert integer to word
    if fit_model == 'gaussian2D':   # if gaussian2D, do the fit with it
        amplitude, center_x, center_y, sigma_x, sigma_y, ellip, center_deg, sn_r = fit_image_2d_gaussian(
            image, ID, high_var_IDs, noisy_IDs, psf_array, image_showing, save_image,
            fitting_method, lya, do_resample=do_resample, resample_coeff=resample_coeff)
    elif fit_model == 'something_else':     # TODO add more fit models here
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
        else:       # TODO add here more model names; e.g.
                    #elif model_array[index, :][1] == 1:
                    #    fit_model = 'SOMETHING_ELSE'
            print("Unknown model, assuming Gaussian 2D")
            fit_model = 'gaussian2D'
    else:
        fit_model = 'gaussian2D'
    return fit_model


def get_fitting_method(ID, fitting_method):
    """
    Takes ID and 2D array of fitting methods. Checks that fitting method exists and returns its name. By default returns
    'least_squares'
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
            method = 'least_squares'
    else:
        method = 'least_squares'
    return method


def fit_image_2d_gaussian(image, ID, high_var_IDs, noisy_IDs, psf_array, image_showing, save_image, fitting_method, lya, do_resample=False, resample_coeff=1):
    """
    Does the fitting using 2D Gaussian model
    :param image: MPDAF image
    :param ID: int ID of the object
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
    :param do_resample: optional, True/False. False by default. If true, it will do resampling
    :param resample_coeff: optional, integer. 1 by defaultt. If do_resample True, then will do resample by the
    coefficient amount
    :return: 7 floats and 1 2D array, parameters of the fit. Order: amplitude, 2D array of center_deg [dec, ra],
    center_x, center_y, ellip, sigma_x, sigma_y, sn_r
    """
    param, sn_r = get_image_params_gaussian2d(image, psf_array, lya, ID, noisy_IDs, high_var_IDs, image_showing,
                                              save_image, fitting_method, do_resample=do_resample,
                                              resample_coeff=resample_coeff)    # fitting is done here
    amplitude, center_x, center_y, sigma_x, sigma_y = convert_params_to_variables(param)    # converts array to separate variables
    ellip = calc_ellip(sigma_x, sigma_y)    # calculates ellipticity
    if do_resample: # if resampling is done, then resample image to correctly calculate centers in ra/dec
        image, _ = resample_image(image, psf_array, resample_coeff)     # overrides image variable if resampled
    center_deg = get_image_ra_dec(image, center_x, center_y)       # calculates center in ra dec from pixel values

    return amplitude, center_x, center_y, sigma_x, sigma_y, ellip, center_deg, sn_r


def fit_image_smth_else(image, ID, high_var_IDs, noisy_IDs, psf_array, image_showing, save_image,
            fitting_method, lya):  # TODO add new fitting methods here
    return 0

def reduce_image(image, cen_x, cen_y, sig_x, sig_y, noisy_IDs, ID, priors = np.zeros(16), do_resample=False, resample_coeff=1):
    """
    Reduces the image size to either minimum size or several sigmas of the galaxy (whichever bigger; adjust values
    in this function). Reduced image is ALWAYS SQUARE.
    :param image: MPDAF image
    :param center_x: Float center_x in pixels
    :param center_y: Float center_y in pixels
    :param sigma_x: Float sigma_x in pixels
    :param sigma_y: Float sigma_y in pixels
    :param noisy_IDs: 2D array with noisy_IDs but only for one specific image (lya or UV). Here does not check which one
    :param ID: ID of the image
    :param priors: Priors for fitting and limits
    :param do_resample: optional, True/False. False by default. If true, it will do resampling
    :param resample_coeff: optional, integer. 1 by defaultt. If do_resample True, then will do resample by the
    coefficient amount
    :return: Reduced image, 2D array with fixes values for noisy_IDs if needed and 1D array of fixed priors
    """
    minimum_image_size_in_pixels = 21   # increase/decrease if need bigger/smaller image.
    min_scaling_of_sigma = 6    # how many sigmas is the galaxy size. 5 seemed to be good enough; 6 used for safety.
    min_size = max(minimum_image_size_in_pixels, max(sig_x, sig_y) * min_scaling_of_sigma)

    subimage = (image.subimage((cen_x, cen_y), min_size, unit_center=None, unit_size=None)).copy()  # creates and
    # copies subimage; original variable image should stay intact

    center_pix = subimage.shape[0] / 2  # subimage is centered, so center in pixels is subimage size / 2
    center_pix_og = center_pix
    if do_resample:     # if resampled, fixes center pixel/sigmas. Resampling of subimage is NOT done here. just fixes values for later fittings
        center_pix = center_pix * resample_coeff
        sig_x = sig_x * resample_coeff
        sig_y = sig_y * resample_coeff

    new_priors = np.copy(priors)
    if ID in noisy_IDs[:, 0]:   # if noisy ID, fixes them to center image
        index = np.where(noisy_IDs[:, 0] == ID)[0][0]
        spec_noisy_ID = noisy_IDs[index, :]
        spec_noisy_ID[1] = center_pix_og
        spec_noisy_ID[2] = center_pix_og
    else:   # othewise returns "empty" array
        spec_noisy_ID = [0, 0, 0, 0, 0]
    for i in range(3, 9):   # fixes the priors to be as a new image
        new_priors = get_prior_array(priors[0], center_pix, center_pix, sig_x, sig_y)
    return subimage, np.array([spec_noisy_ID, [0, 0, 0, 0, 0]]), new_priors

def clean_image(image, ID, noisy_IDs, high_var_IDs, do_resample=False, resample_coeff=1):
    """
    Takes image and if it is marked as noisy image or with high variance, it cleans the image. Otherwise returns
    original data
    :param image: image file
    :param ID: ID of the image
    :param noisy_IDs: IDs of images with high noise
    :param high_var_IDs: 2D array with specific lya/UV IDs of objects with high variance errors
    :return: Returns data 2D array with masked pixels, if they are high variance/noisy (only if mentioned as those IDs)
    """
    data = np.copy(image.data)      # copies, so the original image does not get overriden
    if ID in high_var_IDs[:, 0]:
        index = np.where(high_var_IDs[:, 0] == ID)[0][0]
        data = get_image_with_low_var_errors(image, high_var_IDs[index, :][1])
    if ID in noisy_IDs[:, 0]:
        index = np.where(noisy_IDs[:, 0] == ID)[0][0]
        data = get_clean_image(data, noisy_IDs[index, :], do_resample=do_resample, resample_coeff=resample_coeff)
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
    # image.mask_variance(np.mean(var))     # in theory should work; for some reason never worked.
    # max_var = (0.2 * (np.max(var) - np.min(var)) + np.min(var))
    if max_std_var != 0:    # if specific max tolerance given by user initially
        max_var = np.mean(np.nan_to_num(var, nan=0.0)) + max_std_var * np.std(np.nan_to_num(var, nan=0.0))
    else:   # otherwise use default value, from the beginning of the file
        max_var = np.mean(np.nan_to_num(var, nan=0.0)) + default_std_var * np.std(np.nan_to_num(var, nan=0.0))
    data = np.copy(np.where((var > max_var), masking_value, image.data))    # copies again just in case

    return data

def get_clean_image(data, params, do_resample=False, resample_coeff=1):
    """
    Takes in the noisy image and returns the cleaned image
    :param data: The 2D array data of the image
    :param params: parameters of the image. params[i], with i=0: ID, i=1: center_x, i=2: center_y, i=3: tolerance
    from center from which nothing is cleaned (where galaxy is located), i=4: lya or UV
    :return: 2D cleaned data array
    """
    if do_resample: # if resampling, fixes teh parameters
        cen_x = params[1] * resample_coeff
        cen_y = params[2] * resample_coeff
        tol = params[3] * resample_coeff
    else:   # otherwise just take initial parameters given by the user normally
        cen_x = params[1]
        cen_y = params[2]
        tol = params[3]

    data_size_x = np.size(data[0])
    total_size = np.size(data)
    data_size_y = int(total_size / data_size_x)

    if cen_x <= 0 or cen_y <= 0 or tol <= 0:    # if any user parameters are 0 or negative; try to automatically calculate them
        if cen_x <= 0:      # assumes center of the image is galaxy if not specified
            cen_x = int(data_size_x / 2)
        if cen_y <= 0:      # assumes center of the image is galaxy if not specified
            cen_y = int(data_size_y / 2)

        if tol <= 0:    # assumes the galaxy is within tolerance of the center. if not specified, then within 0.1 radius of the image
            tolr = data_size_x * 0.1
        else:       # otherwise use user value
            tolr = tol

        center_array = np.copy(data[int(cen_y - tolr):int(cen_y + tolr), int(cen_x - tolr):int(cen_x + tolr)])  # center array; the supposed galaxy location

        index_of_cen = np.unravel_index(center_array.argmax(), center_array.shape)  # finds the brightest pixel in center array
        cen_x = index_of_cen[1] + cen_x - tolr  # fixes the center of the galaxy for original image
        cen_y = index_of_cen[0] + cen_y - tolr

        tol = max(int(data_size_x / 4), int(data_size_y / 4))   # new tolerance is 0.25 of the image size

    tol = min(tol, int(data_size_x / 2 - 1), int(data_size_y / 2 - 1))  # takes minimum value (potentially between user) and the size of the image
    galaxy_center = np.copy(data[max(0, int(cen_y-tol)):min(int(cen_y+tol), data_size_y-1), max(0, int(cen_x-tol)):min(int(cen_x+tol), data_size_x-1)]) # center of galaxy is not masked at all
    max_val = np.max(np.nan_to_num(galaxy_center, nan=0.0))     # finds brightest pixel of "galaxy array"
    min_val = np.min(np.nan_to_num(galaxy_center, nan=0.0))

    percentage_tol = 0.2        # tolerance within which pixels are masked. In short pixels > 0.2 * max_galaxy_pixel are masked

    max_tolerance_value = max((max_val - min_val) * percentage_tol, max_val * percentage_tol) + max(0, min_val)   # Adjust if necessary
    replace_by_value = masking_value      # Adjust if necessary

    new_data = np.copy(data)

    new_data = np.where(new_data>max_tolerance_value, replace_by_value, new_data)   # masks here

    new_data[max(0, int(cen_y-tol)):min(int(cen_y+tol), data_size_y-1), max(0, int(cen_x-tol)):min(int(cen_x+tol), data_size_x-1)] = galaxy_center  # galaxy center is returned
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
    """
    Calculates MANY ellipticities of the galaxies, given sigma parameters as arrays. Order of sigmas doesn't matter
    :param sigma_1: 1D array of sigma parameter
    :param sigma_2: 1D array of another sigma parameter
    :return: Returns 1D array of ellipticities of the galaxy
    """
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
    Returns ARRAY of the Ra and Dec coordinates of (x,y) pixels in the image
    :param image: The image file
    :param x_s: x-pixel values
    :param y_s: y-pixel values
    :return: 2D array of [dec, ra] that are dependent on x,y pixels in the same image
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
    
    # 01 Oct 2021: fixed issue with radians/degrees.
    
    ra1 = obj1[1] / 180 * np.pi
    dec1 = obj1[0] / 180 * np.pi
    ra2 = obj2[1] / 180 * np.pi
    dec2 = obj2[0] / 180 * np.pi
    return np.arccos((np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))) / np.pi * 180

def ang_sep_many(arr1, arr2):
    """
    Takes [dec, ra] object positions and returns angular separation between the objects in degrees
    :param arr1: 2D array, each element in format [dec, ra] coordinates of first object
    :param arr2: 2D array, each element in format [dec, ra] coordinates of second object
    :return: 1D array Angular separation between two objects in degrees
    """
    ra1 = arr1[:, 1] / 180 * np.pi
    dec1 = arr1[:, 0] / 180 * np.pi
    ra2 = arr2[:, 1] / 180 * np.pi
    dec2 = arr2[:, 0] / 180 * np.pi
    return np.arccos(np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)) / np.pi * 180

def kpc_sep(ang_sep, z):
    """
    Given angular separation in arcseconds and the redshift z, what is the angular separation between two objects.
    Assumes flat lambdaCDM model with h0 = 0.7, baryon mass = 0.3.
    :param ang_sep: Angular separation between two objects in arcseconds
    :param z: Redshift to objects z
    :return: Float distance between two objects in kpc
    """
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    d_A = cosmo.angular_diameter_distance(z=z)
    theta = ang_sep / 3600  # convert from arcsec to degrees
    theta_radian = theta * np.pi / 180  # convert from degrees to radiangs
    distance_kpc = d_A * theta_radian
    return distance_kpc / u.Mpc * u.kpc * 1000

def kpc_sep_many(ang_sep_arr, z):
    """
    Given angular separation in arcseconds and the redshift z, what is the angular separation between two objects.
    Assumes flat model with h0 = 0.7, baryon mass = 0.3. Can inside arrays as well
    :param ang_sep_arr: Angular separation between two objects in arcseconds array
    :param z: Redshift to objects z
    :return: 1D array of distances between two objects in kpc
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
    given values and returns the arrays with corresponding values. Unsorted array should work, but no promises
    :param values: Values with all stats in the form of "ID, RID, Ra, Dec, z, Ra, Dec". First ra/dec are Lya,
    second ones are UV.
    :param psf_values: 2D array of the values of PSF function. They are in form "ID, alpha, beta, sigma" for PSF
    Kernel
    :param IDs_to_find: Which IDs to find among the values
    :param start_at_ID: If specified, the array will start at that specific ID. Good if you need to do calculations
    in several steps and need to restart at specific ID.
    :return: Returns 7 1D arrays in the same order as values ("ID, RID, Ra, Dec, z, Ra, Dec"). Those represent
    ID with corresponding values.
    """
    index_to_start = 0
    if start_at_ID > 0:
        index_to_start = np.where(values[:,0].astype(int) == start_at_ID)[0]
        if index_to_start:  # if start_at_ID does not exist in array, ignore it
            index_to_start = index_to_start[0]
        else:
            index_to_start = 0  # default start at 0

    values = values[values[:, 0].argsort()]     # should sort arrays, so they are OK even if unsorted
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
    "ID, Isolated, Centered, 3rd panel OK, HST exists". Kinda ignores last 2 columns for now; change if needed
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
    """
    Loads text file with HST IDs
    :return: returns values with HST IDs
    """
    return np.loadtxt(filepath_HST_IDs).astype(int)

def get_no_HST_IDs():
    """
    Loads text file with ONLY Lya IDs
    :return: returns values with ONLY Lya IDs
    """
    return np.loadtxt(filepath_no_HST_IDs).astype(int)

def get_uncertainties(ID, image, psf_array, number_of_slices, noisy_IDs, high_var_IDs, model, fitting_method, lya, priors=np.zeros(16), do_resample=False, resample_coeff=1):
    """
    Gets UV, Lya image and creates number_of_slices images and calculates uncertainties
    :param ID: ID that is used (for naming purposes only)
    :param image: Image data of lya
    :param psf_array: 1D array with PSF parameters. psf_array[j] gives 1D array of specific values. j=0: ID, j=1: alpha, j=2: beta,
    j=3: sigma of PSF.
    :param number_of_slices: How many images to test (1000 should be standard, but slow)
        :param noisy_IDs: 2D array with IDs that need to be masked. Format: ID, center_x (in pixels), center_y
    (in pixels), tolerance (in pixels, within tolerance no masking), lya (1 is lya, 0 is uv)
    :param high_var_IDs: 2D array with IDs that need high variance pixels to be masked. Format: ID,
    variance tolerance (float; variance with var_mean + var_tol * var_st.dev are masked), lya (1 is lya, 0 is uv)
    :param model: string, model name (e.g. gaussian2D)
    :param fitting_method: fitting method, if not default
    :param lya: 1 is lya, 0 is UV
    :param priors: the 1D array with priors for simpler fitting
    :param do_resample: optional, True/False. False by default. If true, it will do resampling
    :param resample_coeff: optional, integer. 1 by defaultt. If do_resample True, then will do resample by the
    coefficient amount
    :return: Returns 1D array with uncertainties in the following order:
    [amp_lya, cen_x_lya, cen_y_lya, sig_x_lya, sig_y_lya, ellip_lya, amp_uv, cen_x_uv, cen_y_uv, sig_x_uv, sig_y_uv,
    ellip_uv, ang_separation, kpc_separation]
    """
    if do_resample:     # resamples image, if resampling needed
        image, psf_array = resample_image(image, psf_array, resample_coeff)

    amplitudes, center_x_s, center_y_s, cube_boot, ellipticity, error, sigma_x_s, sigma_y_s, x, y = prepare_uncert_calcs(
        image, number_of_slices, ID, noisy_IDs, high_var_IDs, lya, do_resample=do_resample, resample_coeff=resample_coeff)
    # creates the empty variables where to save values, prepares cubes and prepares x/y/error for fits

    if lya: # purely for naming
        name = "Lya"
    else:
        name = "UV"

    if number_of_slices >= threshold_for_workers and not debug_mode:    # if multithreading, do this
        amplitudes = np.array([])
        center_x_s = np.array([])
        center_y_s = np.array([])
        ellipticity = np.array([])
        sigma_x_s = np.array([])
        sigma_y_s = np.array([])
        center_deg = np.array([[]])
        futures = []
    else:   # otherwise just do normal
        center_deg = np.zeros((number_of_slices, 2))

    if model == "gaussian2D":   # TODO add more models here
        if number_of_slices >= threshold_for_workers and not debug_mode:    # multithreading here
            print("starting uncertainty calculations")
            for im_boot in cube_boot:   # submits the work to the clients
                future = client.submit(gauss2d_fitting_no_plot_params, im_boot.data, x, y, error, psf_array, lya, fitting_method, priors)
                futures.append(future)  # prepares to get values

            print("start gathering")    # use http://localhost:8787/status to check status. the port might be different
            futures = np.array(client.gather(futures))  # starts the calculations (takes a long time here)
            amplitudes = futures[:, 0, 0]
            center_x_s = futures[:, 0, 1]
            center_y_s = futures[:, 0, 2]
            sigma_x_s = futures[:, 0, 3]
            sigma_y_s = futures[:, 0, 4]
            ellipticity = calc_ellip_many(sigma_x_s, sigma_y_s)
            center_deg = get_image_ra_dec_many(image, center_x_s, center_y_s)
            print("Worker calculation done")    # when done, save values
        else:   # if not multithreading, do using 1 core. slower, but better for debugging
            for j in range(0, number_of_slices):
                im_boot = cube_boot[j]
                parameters = gauss2d_fitting_no_plot_params_img(im_boot.data, x, y, image.var, psf_array, lya, debug_mode, fitting_method, priors=priors)
                # saves images, if debug_mode. otherwise, just do calculations

                amplitudes[j], center_x_s[j], center_y_s[j], sigma_x_s[j], sigma_y_s[j] = convert_params_to_variables(parameters)
                ellipticity[j] = calc_ellip(sigma_x_s[j], sigma_y_s[j])
                center_deg[j] = get_image_ra_dec(image, center_x_s[j], center_y_s[j])

                if j % 20 == 0:
                   print(int(j / number_of_slices * 100), "% done for image", name, ID)

    if do_resample: # if resampling, fix centers, sigmas
        center_x_s = center_x_s / resample_coeff
        center_y_s = center_y_s / resample_coeff
        sigma_x_s = sigma_x_s / resample_coeff
        sigma_y_s = sigma_y_s / resample_coeff

    return [np.std(amplitudes), np.std(center_x_s), np.std(center_y_s), np.std(sigma_x_s),
            np.std(sigma_y_s), np.std(ellipticity), center_deg]  # returns st.dev. (uncertainties) and center


def prepare_uncert_calcs(image, number_of_slices, ID, noisy_IDs, high_var_IDs, lya, do_resample=False, resample_coeff=1):
    """
    Prepares the number_of_slices cubes for the uncertainty calculations
    :param image: image MPDAF file
    :param number_of_slices: number of slices for uncertainty calculations
    :param ID: ID of object
    :param noisy_IDs: IDs of noisy objects
    :param high_var_IDs: 2D array with IDs of objects with high variance errors
    :param lya: True for lya, False for UV
    :param do_resample: optional, True/False. False by default. If true, it will do resampling
    :param resample_coeff: optional, integer. 1 by defaultt. If do_resample True, then will do resample by the
    coefficient amount
    :return: Returns 10 lists: empty amplitude, center_x, center_y, cube_boot (full of cubes randomly changed within
    variance for uncertainty calculations), empty ellipticity, flattened error, empty sigma_x, sigma_y, flattened x, y
    """
    imlist = []
    if lya: # cleans images if needed
        data = clean_image(image, ID, noisy_IDs[noisy_IDs[:, 4] == 1], high_var_IDs[high_var_IDs[:, 2] == 2], do_resample=do_resample, resample_coeff=resample_coeff)
    else:
        data = clean_image(image, ID, noisy_IDs[noisy_IDs[:, 4] == 0], high_var_IDs[high_var_IDs[:, 2] == 2], do_resample=do_resample, resample_coeff=resample_coeff)
    for d, v in zip(data.ravel(), image.var.ravel()):   # creates many cubes, where each pixel is varied within normal distribution of its error
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





def get_image_params_gaussian2d(image, psf_array, lya, ID, noisy_IDs, high_var_IDs, image_showing, save_image, fitting_method, do_resample=False, resample_coeff=1):
    """
    Takes image and variables and returns the parameters of the Gaussian2D fit. Basically, fitting done here
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
    :param do_resample: optional, True/False. False by default. If true, it will do resampling
    :param resample_coeff: optional, integer. 1 by defaultt. If do_resample True, then will do resample by the
    coefficient amount
    :return: Parameters tuple with parameters of the Gaussian2D fit.
    """
    if do_resample: # if resampling
        image, psf_array = resample_image(image, psf_array, resample_coeff)

    if lya: # cleans image
        data = clean_image(image, ID, noisy_IDs[noisy_IDs[:, 4] == 1], high_var_IDs[high_var_IDs[:, 2] == 1], do_resample=do_resample, resample_coeff=resample_coeff)
    else:
        data = clean_image(image, ID, noisy_IDs[noisy_IDs[:, 4] == 0], high_var_IDs[high_var_IDs[:, 2] == 0], do_resample=do_resample, resample_coeff=resample_coeff)
    x, y, error = flatten_xy_err(data, image.var) # creates flat x,y, error

    method = get_fitting_method(ID, fitting_method) # gets method if needed

    out, mod = gauss2d_fitting_no_plot(data, x, y, error, psf_array, lya, fitting_method=method)    # does fitting here

    if image_showing or save_image: # if needed, shows or saves image
        if lya:
            name = "Lya_of_" + str(ID)
        else:
            name = "UV_of_" + str(ID)
        plot_image(out, data, x, y, image.var, mod, name, image_showing, save_image)
    params = out.params

    #save_redchi(ID, lya, resample_coeff, out.redchi) # if needed, uncomment for the reduced chi-squared

    sn_r = get_sn(data, out, mod, x, y) # signal-to-noise-ratio estimate
    return params, sn_r


def save_redchi(ID, lya, resample_coeff, redchi):
    """
    Saves reduced chi-squared. Temporary function
    :param ID: ID
    :param lya: Lya: 1, UV: 0
    :param resample_coeff: Coefficient of resampling
    :param redchi: The reduced chi-squared
    :return: Nothing, saves as text file
    """
    save_in_txt_topcat([ID, lya, resample_coeff, redchi], "redchi.txt")



def resample_image(image, psf_array, resample_coeff):
    """
    Resamples image (copies it to not change original image).
    :param image: MPDAF image
    :param psf_array: PSF array, to adjust values for PSF as well
    :param resample_coeff: int, by how much to resample
    :return: Resampled image and resampled psf_array
    """
    new_shape = (np.asarray(image.shape) * resample_coeff).astype(int)
    new_step = image.get_step(unit=u.arcsec) / resample_coeff
    resampled_image = image.resample(newdim=new_shape, newstart=None, newstep=new_step, unit_step=u.arcsec, flux=False)
    image = resampled_image.copy()
    psf_array = np.copy(psf_array)
    psf_array[1] = psf_array[1] * resample_coeff    # only 2 parameters resampled
    psf_array[3] = psf_array[3] * resample_coeff
    return image, psf_array


def get_sn(og_data, out, mod, x, y):
    """
    Gets estimate of signal to noise ratio according to the https://en.wikipedia.org/wiki/Signal-to-noise_ratio#Definition,
    where SNR = mean(signal^2) / mean(residuals^2). Might require adjustments
    :param og_data: 2D array of original data
    :param out: out from fit
    :param mod: which mod has been used (e.g. gaussian)
    :param x: 1D array of x coordinates
    :param y: 1D array of y coordinates
    :return: SNR float value estimate
    """
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
    OLD ONE FROM SCIPY

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
