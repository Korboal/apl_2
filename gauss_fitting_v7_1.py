import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters, Parameter
from scipy.interpolate import griddata
import datetime
from astropy.convolution import kernels
from scipy.signal import convolve as sciconv

default_fit_met_gauss2d = 'least_squares'   # default fitting method for gaussian2D

def gaussian2D(x, y, centerx, centery, sigmax, sigmay, amplitude, theta):  # TODO: find a way to optimise? Spending half of total time here because of convolution
    """
    Gaussian 2D Model, also does convolution with global kernel parameter
    :param x: x values
    :param y: y values
    :param centerx: center x of the gaussian
    :param centery: center y of the gaussian
    :param sigmax: sigma x of gaussian
    :param sigmay: sigma y of gaussian
    :param amplitude: ampplitude of gaussian
    :return: array with image fit
    """
    cost2 = np.cos(theta) ** 2
    sint2 = np.sin(theta) ** 2
    sin2t = np.sin(2. * theta)
    xstd2 = sigmax ** 2
    ystd2 = sigmay ** 2
    xdiff = x - centerx
    ydiff = y - centery
    a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2))
    b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2))
    c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2))
    model = amplitude * np.exp(-((a * xdiff ** 2) + (b * xdiff * ydiff) + (c * ydiff ** 2)))    # Gaussian 2D with rotation

    #model = (amplitude * np.exp(-(((centerx - x) / sigmax) ** 2 + ((centery - y) / sigmay) ** 2) / 2.0)) # Gaussian 2D without rotation

    if np.size(model[0]) == np.size(kernel[0]):  # adjust kernel, if not same size
        to_conv = kernel
    else:
        to_conv = kernel.flatten()
    values = sciconv(model, to_conv, mode='same') # convole with the kernel; astropy convolution seemed slower, using scipy here
    return values

def gauss2d_fitting_no_plot(data, x, y, error, psf_array, lya, fitting_method=default_fit_met_gauss2d, priors=np.zeros(16)):
    """
    Does the fitting, does not plot results
    :param data: takes 2D data of the original MPDAF image
    :param x: x-values 1D array calculated from xyz flatten
    :param y: y-values 1D array calculated from xyz flatten
    :param error: the error 1D array calculated from xyz flatten
    :param psf_array: 1D array with PSF parameters. psf_array[j] gives specific values. j=0: ID, j=1: alpha, j=2: beta, j=3: sigma of PSF.
    :param lya: Boolean. True if Lya, False if UV. used for creation of PSF
    :param fitting_method: optional, if non-default fitting method.
    :param priors: optional, priors to guess and limit the fit; otherwise auto creates limits and guesses
    :return: returns the z-fit values
    """
    z = flatten_z(data) # flattes to 1D

    mod = get_model()  # gets the model name (gaussian2D right now)

    global kernel       # global, because that was easiest way to solve some other problems
    kernel_x_size = np.max(x) + 1  # should always be odd
    kernel_y_size = np.max(y) + 1

    if lya:     # create PSF kernel. For Lya and UV different types of Kenerl
        kernel = kernels.Moffat2DKernel(psf_array[1], psf_array[2], x_size=kernel_x_size, y_size=kernel_y_size).array
    else:
        kernel = kernels.Gaussian2DKernel(psf_array[3], x_size=kernel_x_size, y_size=kernel_y_size).array
    pars = get_param_guess(x, y, z, priors=priors)  # Make a guess or use priors
    out = mod.fit(z, params=pars, x=x, y=y, weights=1 / error, method=fitting_method)   # do the fit
    #out = mod.fit(z, params=pars, x=x, y=y, weights=1 / error, method='least_squares')  # 'least_squares'

    return out, mod

def gauss2d_fitting_no_plot_params_img(data, x, y, error, psf_array, lya, save_image, fitting_method=default_fit_met_gauss2d, priors=np.zeros(16)):
    """
    Does the fit and if save_image is True, then saves the image
    :param data: takes 2D data of the original MPDAF image
    :param x: x-values 1D array calculated from xyz flatten
    :param y: y-values 1D array calculated from xyz flatten
    :param error: the error 1D array calculated from xyz flatten
    :param psf_array: 1D array with PSF parameters. psf_array[j] gives specific values. j=0: ID, j=1: alpha, j=2: beta, j=3: sigma of PSF.
    :param lya: Boolean. True if Lya, False if UV. used for creation of PSF
    :param save_image: True/False, whether to save image or not (used for uncertainty debugging)
    :param fitting_method: optional, if non-default fitting method.
    :param priors: optional, priors to guess and limit the fit; otherwise auto creates limits and guesses
    :return: out.params
    """
    out, mod = gauss2d_fitting_no_plot(data, x, y, flatten_error(error), psf_array, lya, fitting_method=fitting_method, priors=priors)
    if save_image:
        if lya:
            name = 'lya'
        else:
            name = 'uv'
        plot_image(out, data, x, y, error, mod, name, False, True)
    return out.params

def gauss2d_fitting_no_plot_params(data, x, y, error, psf_array, lya, fitting_method=default_fit_met_gauss2d, priors=np.zeros(16)):
    """
    Returns the parameters as an array
    :param data: takes 2D data of the original MPDAF image
    :param x: x-values 1D array calculated from xyz flatten
    :param y: y-values 1D array calculated from xyz flatten
    :param error: the error 1D array calculated from xyz flatten
    :param psf_array: 1D array with PSF parameters. psf_array[j] gives specific values. j=0: ID, j=1: alpha, j=2: beta, j=3: sigma of PSF.
    :param lya: Boolean. True if Lya, False if UV. used for creation of PSF
    :param fitting_method: optional, if non-default fitting method.
    :param priors: optional, priors to guess and limit the fit; otherwise auto creates limits and guesses
    :return: 1D array of parameters
    """
    out, mod = gauss2d_fitting_no_plot(data, x, y, error, psf_array, lya, fitting_method=fitting_method, priors=priors)
    return [convert_params_to_variables(out.params)]



def flatten_z(data):
    """
    Flattens data and converts nan to 0.0 for fitting
    :param data: 2D array of data of image
    :return: 1D array of flatten data with nan replaces by 0.0
    """
    im_data = convert_data_to_odd_axes(data)
    try:
        output = (im_data.filled(0.0)).flatten()
    except:
        output = im_data.flatten()
    output = np.nan_to_num(output, nan=0.0)
    return output

def flatten_error(var):
    """
    Flattens variance and converts to error
    :param var: 2D array of variance
    :return: Flattens the errors and replaces 0 and nan by mean of error
    """
    im_var = convert_data_to_odd_axes(var)
    try:
        output = (im_var.filled(0.0)).flatten()
    except:
        output = im_var.flatten()
    error = np.sqrt(output)     # error is st.dev., so take square root of variance to get it
    error = np.nan_to_num(error, nan=-0.001)    # if any nan values, convert to specific small negative value
    error = np.where(error < 0.0000001, np.mean(error), error)  # all values 0 or less are converted to mean error
    # otherwise fitting does not work with nan values. For weightings, I decided to use mean error to give average
    # weight to masked pixels
    return error

def flatten_z_keep_nan(data):
    """
    Converts data 2D array, but does not convert nan to anything (in theory)
    :param data: 2D array of daata
    :return: flattened data
    """
    im_data = convert_data_to_odd_axes(data)
    try:
        output = (im_data.filled(0.0)).flatten() # might convert masked to 0.0? not sure, worked fine for my usage
    except:
        output = im_data.flatten()
    return output

def get_model():
    """
    Returns 2DGaussian model
    :return: Model Gaussian 2D
    """
    return Model(gaussian2D, independent_vars=['x', 'y'])
    #return Gaussian2dModel()

def get_param_guess(x, y, z, priors=np.zeros(16)):
    """
    Returns parameters guess. If priors are given (non 0), then takes them as guess. Otherwise makes its own guess
    :param x: 1D array of x-values
    :param y: 1D array of y-values
    :param z: 1D array of z-values (flattened data)
    :param priors: If priors are known. 0-s changes to default values. Indices:
    0 - amp
    1 - min amp
    2 - max amp
    3 - cen x
    4 - min cen x
    5 - max cen x
    6 - cen y
    7 - min cen y
    8 - max cen y
    9 - sig x
    10 - min sig x
    11 - max sig x
    12 - sig y
    13 - min sig y
    14 - max sig y
    15 - init rotation
    :return: parameters as guess
    """
    priors_to_send = np.zeros(16)

    maxx, minx = np.max(x) + 1, np.min(x)
    maxy, miny = np.max(y) + 1, np.min(y)
    maxz, minz = np.max(z), 0

    centerx = x[np.argmax(z)]
    centery = y[np.argmax(z)]
    height = (maxz - minz)
    sigmax = (maxx - minx) / 6.0
    sigmay = (maxy - miny) / 6.0

    amp = height * sigmax * sigmay
    min_amp = 0
    max_amp = np.inf
    min_cen_x = 0
    max_cen_x = maxx
    min_cen_y = 0
    max_cen_y = maxy

    min_sig_x = 0
    max_sig_x = np.inf
    min_sig_y = 0
    max_sig_y = np.inf
    init_rot = 0

    new_guesses = np.array([amp, min_amp, max_amp, centerx, min_cen_x, max_cen_x, centery, min_cen_y, max_cen_y, sigmax, min_sig_x, max_sig_x, sigmay, min_sig_y, max_sig_y, init_rot])
    for i in range(np.size(priors_to_send)):
        if priors[i] == 0:
            priors_to_send[i] = new_guesses[i]
        else:
            priors_to_send[i] = priors[i]

    pars = get_fit_parameters(*priors_to_send)

    return pars




def get_fit_parameters(amp, min_amp, max_amp, centerx, min_cen_x, max_cen_x, centery, min_cen_y, max_cen_y, sigmax, min_sig_x, max_sig_x, sigmay, min_sig_y, max_sig_y, init_rot):
    """
    Creates parameters based on values given with limits.
    :param amp:
    :param min_amp:
    :param max_amp:
    :param centerx:
    :param min_cen_x:
    :param max_cen_x:
    :param centery:
    :param min_cen_y:
    :param max_cen_y:
    :param sigmax:
    :param min_sig_x:
    :param max_sig_x:
    :param sigmay:
    :param min_sig_y:
    :param max_sig_y:
    :param init_rot:
    :return:
    """
    pars = Parameters()
    a = Parameter('amplitude', value=amp, min=min_amp, max=max_amp)
    cx = Parameter('centerx', value=centerx, min=min_cen_x, max=max_cen_x)
    cy = Parameter('centery', value=centery, min=min_cen_y, max=max_cen_y)
    sx = Parameter('sigmax', value=sigmax, min=min_sig_x, max=max_sig_x)
    sy = Parameter('sigmay', value=sigmay, min=min_sig_y, max=max_sig_y)
    th = Parameter('theta', value=init_rot, min=-np.pi/2, max=np.pi/2)
    pars.add_many(a, cx, cy, sx, sy, th)
    return pars


def is_even(number):
    """
    Checks whether a number is even
    :param number: integer to check
    :return: True if even, False if odd
    """
    return number % 2 == 0

def convert_data_to_odd_axes(og_im_data):
    """
    Converts 2D array to 2D array with odd axes for the convolution to work
    :param og_im_data: 2D array (can be either even or odd axes, not necessarily symmetric)
    :return: 2D array with all odd axes (last row/column is removed if necessary)
    """
    im_data = np.copy(og_im_data)
    if is_even(np.size(im_data[0])):
        im_data = im_data[:, 0:-1]
    if is_even(np.size(im_data) / np.size(im_data[0])):
        im_data = im_data[0:-1, :]
    return im_data

def flatten_xy_err(og_im_data, var):
    """
    Flattens data, variance and returns x, y and error coordinates
    :param og_im_data: the data of the image as 2D array, giving values from lowest left corner (x=0, y=0),
    first going along the x-axis, then each new array is a values along the new y-axis. Both rectangular and
    square images work.
    :param var: 2D variance of the image
    :return: x, y, error 1D arrays. For each x[i] and y[i] corresponds to the correct z[i] value, as it was in
    the image_data with corresponding error
    """

    im_data = convert_data_to_odd_axes(og_im_data)

    data_size_x = np.size(im_data[0])
    total_size = np.size(im_data)
    data_size_y = int(total_size / data_size_x)
    error = flatten_error(var)

    x = np.linspace(0, data_size_x-1, data_size_x)
    x = np.tile(x, data_size_y)
    y = np.linspace(0, data_size_y-1, data_size_y)
    y = np.repeat(y, data_size_x)

    return x, y, error

def plot_image(out, data, x, y, var, mod, name, image_showing, save_image):
    """
    Plots image, given x, y, data and fit in out
    :param out: the fit that was gotten
    :param data: takes image into it and converts into data already inside the function
    :param name: "Lya" or "UV" name for image saving
    :param psf_array: 1D array with PSF parameters. psf_array[j] gives specific values. j=0: ID, j=1: alpha, j=2: beta, j=3: sigma of PSF.
    :param image_showing: Boolean, whether to show fitting images or not.
    :param save_image: Boolean, whether to automatically save fitting images into a folder or not
    :return: prints out the Gauss2Dmodel fitting for the image data and plots OG image, fit, residuals and
    (data-fit) / sigma, where sigma = sqrt(variance)
    """

    x_max = int(np.max(x)+1)
    y_max = int(np.max(y)+1)

    print(out.fit_report())

    X, Y = np.meshgrid(np.linspace(np.min(x), np.max(y), x_max),      # Converts x,y,z values to meshgrid for drawing
                       np.linspace(np.min(y), np.max(y), y_max))
    Z = griddata((x, y), convert_data_to_odd_axes(data).flatten(), (X, Y), method='linear', fill_value=0)
    #Z_og = griddata((x, y), convert_data_to_odd_axes(og_data).flatten(), (X, Y), method='linear', fill_value=0)
    #fig, axs = plt.subplots(2, 3, figsize=(11, 11))       # Draws 4 plots. Data, fit and residuals, residuals/sigma
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # Draws 4 plots. Data, fit and residuals, residuals/sigma
    vmax = np.nanpercentile(data, 99.9)

    #ax = axs[0, 0]
    #art = ax.pcolor(X, Y, Z_og, vmin=0, vmax=vmax, shading='auto')
    #plt.colorbar(art, ax=ax, label='z')
    #ax.set_title('Original data of ' + name)

    ax = axs[0, 0]
    #art = ax.pcolor(X, Y, Z, vmin=0, vmax=vmax, shading='auto')
    art = ax.pcolor(X, Y, Z, vmin=0, shading='auto')
    plt.colorbar(art, ax=ax, label='z')
    ax.set_title('Data of ' + name)

    ax = axs[0, 1]
    fit = mod.func(X, Y, **out.best_values)
    #art = ax.pcolor(X, Y, fit, vmin=0, vmax=vmax, shading='auto')
    art = ax.pcolor(X, Y, fit, vmin=0, shading='auto')
    plt.colorbar(art, ax=ax, label='z')
    ax.set_title('Fit')

    ax = axs[1, 0]
    fit = mod.func(X, Y, **out.best_values)
    #art = ax.pcolor(X, Y, Z-fit, vmin=0, vmax=vmax, shading='auto')
    art = ax.pcolor(X, Y, Z - fit, vmin=0, shading='auto')
    plt.colorbar(art, ax=ax, label='z')
    ax.set_title('Data - Fit')

    ax = axs[1, 1]
    fit = mod.func(X, Y, **out.best_values)
    art = ax.pcolor(X, Y, (Z - fit) / np.sqrt(convert_data_to_odd_axes(var)), vmin=0, shading='auto')
    plt.colorbar(art, ax=ax, label='z')
    ax.set_title('(Data - Fit) / sigma')
    """
    ax = axs[1, 2]
    art = ax.pcolor(X, Y, np.sqrt(var), vmin=0, shading='auto')
    plt.colorbar(art, ax=ax, label='z')
    ax.set_title('Sigma')"""

    for ax in axs.ravel():
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    if save_image:
        image_filename = "output_pictures/" + name + "__" + str(datetime.datetime.now()).replace(':', '_') + ".png"
        plt.savefig( image_filename)

    if image_showing:
        plt.show()

    plt.close()


def convert_params_to_variables(params):
    """
    Converts parameters into separate float variables
    :param params: Tuple with parameters
    :return: 5 floats in order: amplitude, center_x, center_y, sigma_x, sigma_y
    """
    amplitude = params['amplitude'] * 1
    center_x = params['centerx'] * 1
    center_y = params['centery'] * 1
    sigma_x = params['sigmax'] * 1
    sigma_y = params['sigmay'] * 1
    return amplitude, center_x, center_y, sigma_x, sigma_y