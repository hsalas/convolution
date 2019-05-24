# -*- coding: utf-8 -*-
# Author: Héctor Salas

import numpy as np
from astropy.io import fits
from scipy.ndimage import zoom
from astropy.convolution import convolve_fft
import glob

import readline
readline.parse_and_bind('tab: complete')
readline.set_completer_delims(' \t\n')


def load_fits(name):
    """ Return the header and data from a fits file
    Inputs:
        name: name of the .fits file (str).
    Output:
        data: file data (numpy.ndarry)
        header: file header (astropy.io.fits.header.Header)
    """
    while True:
        try:
            data = fits.getdata(name)
            header = fits.getheader(name)
            return data, header
        except FileNotFoundError:
            print(f"File {name} not found")
            name = input('Please enter a different file name: ')


def find_pixel_scale(header):
    """Finds the value of the image pixel scale from the image headers
    Inputs:
        header: Header of the image
    Output:
        pixel_scale: Pixel scale of the image in arcsec/pixel
    """
    
    pixel_scale = None
    keys = [key for key in header.keys()]

    if ('CD1_1' in keys) and ('CD1_2' in keys):
        pixel_scale = np.sqrt(header['CD1_1']**2 + header['CD1_2']**2)*3600

    elif ('PC1_1' in keys) and ('PC1_2' in keys):
        pixel_scale = np.sqrt(header['PC1_1']**2 + header['PC1_2']**2)*3600

    elif 'PXSCAL_1' in keys:
        pixel_scale = abs(header['PXSCAL_1'])

    elif 'PIXSCALE' in keys:
        pixel_scale = header['PIXSCALE']

    elif 'SECPIX' in keys:
        pixel_scale = header['SECPIX']

    elif 'CDELT1' in keys:
        pixel_scale = abs(header['CDELT1'])*3600

    else:
        print('Unable to get pixel scale from image header')
        while True:
            pixel_scale = input('Plesae input the pixel scale value in \
                                arcsec per pixel')
            try:
                pixel_scale = float(pixel_scale)
                return pixel_scale
            except ValueError:
                pass

    return pixel_scale


def save_fits(name, data, header):
    """Saves data to a fits file

    input:
        name: name of file to save (str).
        data: data to saved
    """
    name = name[:-5] + '_convolved.fits'
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    # check if file already exists, if tru ask if overwrite file or not
    aux = glob.os.path.isfile(name)
    while aux:
        print(f'File {name} already exists.')
        # loop to force user to enter only yes or no options
        while True:
            ow = input('Overwrite file (yes/no)? ')
            ow.lower()
            try:
                aux = valid[ow]
                break
            except ValueError:
                pass
        if not aux:
            name = input('Enter new name to save the convolved image: ')
            # check if new name chosen already exists
            aux = glob.os.path.isfile(name)
        else:
            aux = False
    fits.writeto(name, data, header=header, overwrite=True)
    print(f'convolved image saved as {name}')

#complete this function to get a more invormative header
def update_header(header_i, header_k):
    header = header_i.copy()
    header['history'] = ('Convolved version of created with convolve_images.py')
    return header

def do_the_convolution(image, image_h, kernel, kernel_h):
    """Function that convolve an image with a kernel

    Inputs:

        image:  image data
        image_h: image header
        kernel: kernel data
        kernel_h: kernel_header

    Outputs:

    """

    # Get image and kernel pixel_scale
    pixel_scale_i = find_pixel_scale(image_h)
    pixel_scale_k = find_pixel_scale(kernel_h)
    # resize kernel if necessary
    if pixel_scale_k != pixel_scale_i:
        ratio = pixel_scale_k / pixel_scale_i
        size = ratio*kernel.shape[0]
        # ensure a odd kernel
        if round(size) % 2 == 0:
            size += 1
            ratio = size / kernel.shape[0]
        kernel = zoom(kernel, ratio) / ratio**2
        
    # do convolution
    if len(np.shape(image)) == 2:
        convolved = convolve_fft(image, kernel, nan_treatment='interpolate',
                             normalize_kernel=True, preserve_nan=True,
                             boundary='fill', fill_value=0., allow_huge=True)
    else:
        convolved = []
        for i in range(len(np.shape(image))+1):
            convolved_i = convolve_fft(image[i], kernel,
                                       nan_treatment='interpolate',
                                       normalize_kernel=True,
                                       preserve_nan=True, fft_pad=True,
                                       boundary='fill', fill_value=0., allow_huge=True)
            convolved.append(list(convolved_i))
        convolved = np.asarray(convolved)

    return convolved, kernel


if __name__ == '__main__':
    # load kernel
    kernel_name = input('Please enter kernel name: ')
    kernel, header_k = load_fits(kernel_name)
    # load image
    image_name = input('Please enter image name: ')
    image, header_i = load_fits(image_name)
    # do convolution
    convolved, kernel = do_the_convolution(image, header_i, kernel, header_k)
    #update the header info
    #function not finished
    header_new = update_header(header_i, header_k)
    # save convolved image and resized kerne
    save_fits(image_name, convolved, header_new)
