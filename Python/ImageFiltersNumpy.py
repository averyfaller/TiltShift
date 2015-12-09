import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import color
import time
import math
import argparse

# A basic Python implementation of ImageFilters

def numpy_boxblur(image, blur_mask, iterations=3):
    ''' boxblur using numpy '''
    
    blur_mask = blur_mask.astype(np.float)
    self_blur_mask = (9 - (blur_mask * 8)) / 9.0
    other_blur_mask = blur_mask / 9.0
    
    red = image[...,0].astype(np.float)
    green = image[...,1].astype(np.float)
    blue = image[...,2].astype(np.float)
    
    blur_weights = np.dstack((other_blur_mask, other_blur_mask, other_blur_mask,
                       other_blur_mask, self_blur_mask, other_blur_mask,
                       other_blur_mask, other_blur_mask, other_blur_mask))    
    iterations = 1
    
    print blur_weights[200,300,:]
    
    for i in range(iterations):
        red_padded = np.pad(red, 1, mode='edge')
        green_padded = np.pad(green, 1, mode='edge')
        blue_padded = np.pad(blue, 1, mode='edge')
        
        red_stacked = np.dstack((red_padded[:-2,  :-2], red_padded[:-2,  1:-1], red_padded[:-2,  2:],
                                 red_padded[1:-1, :-2], red_padded[1:-1, 1:-1], red_padded[1:-1, 2:],
                                 red_padded[2:,   :-2], red_padded[2:,   1:-1], red_padded[2:,   2:]))
        green_stacked = np.dstack((green_padded[:-2,  :-2], green_padded[:-2,  1:-1], green_padded[:-2,  2:],
                                   green_padded[1:-1, :-2], green_padded[1:-1, 1:-1], green_padded[1:-1, 2:],
                                   green_padded[2:,   :-2], green_padded[2:,   1:-1], green_padded[2:,   2:])) 
        blue_stacked = np.dstack((blue_padded[:-2,  :-2], blue_padded[:-2,  1:-1], blue_padded[:-2,  2:],
                                  blue_padded[1:-1, :-2], blue_padded[1:-1, 1:-1], blue_padded[1:-1, 2:],
                                  blue_padded[2:,   :-2], blue_padded[2:,   1:-1], blue_padded[2:,   2:])) 
        
        red = np.average(red_stacked, axis=2, weights=blur_weights).astype(np.uint8)
        green = np.average(green_stacked, axis=2, weights=blur_weights).astype(np.uint8)
        blue = np.average(blue_stacked, axis=2, weights=blur_weights).astype(np.uint8)

    image = np.dstack((red, green, blue))
    return image


# A method that takes in a matrix of 3x3 pixels and blurs
# the center pixel based on the surrounding pixels, a
# bluramount of 1 is full blur and will weight the neighboring
# pixels equally with the pixel that is being modified.
# While a bluramount of 0 will result in no blurring.
def boxblur(blur_amount, p0, p1, p2, p3, p4, p5, p6, p7, p8):
    # Calculate the blur amount for the central and
    # neighboring pixels
    self_blur_amount = (9 - (blur_amount * 8)) / 9.0
    other_blur_amount = blur_amount / 9.0

    # Sum a weighted average of self and others based on the blur amount
    #print "--------"
    #print 0 + p0[0] + p1[0] + p2[0] + p3[0] + p5[0] + p6[0] + p7[0] + p8[0]

    red_v = (self_blur_amount * p4[0]) + (other_blur_amount * (0 + p0[0] + p1[0] + p2[0] + p3[0] + p5[0] + p6[0] + p7[0] + p8[0]))
    green_v = (self_blur_amount * p4[1]) + (other_blur_amount * (0 + p0[1] + p1[1] + p2[1] + p3[1] + p5[1] + p6[1] + p7[1] + p8[1]))
    blue_v = (self_blur_amount * p4[2]) + (other_blur_amount * (0 + p0[2] + p1[2] + p2[2] + p3[2] + p5[2] + p6[2] + p7[2] + p8[2]))
    #print "Original %s" % p4
    #print "New %s" % [int(red_v), int(blue_v), int(green_v)]
    #print "--------"
    return [int(red_v), int(green_v), int(blue_v)]

# Adjusts the brightness on a pixel
def brightness(p,value):
    red = truncate(p[0] + value)
    green = truncate(p[1] + value)
    blue = truncate(p[2]+ value)

    return [red, green, blue]

# Adjusts the saturation of a pixel
# 0.0 creates a black-and-white image.
# 0.5 reduces the color saturation by half.
# 1.0 causes no change.
# 2.0 doubles the color saturation.
def saturation(p, value):
    # red_v = p[0] * (1 - value)
    # blue_v = p[1] * (1 - value)
    # green_v = p[2] * (1 - value)
    Pr = 0.299
    Pg = 0.587
    Pb= 0.114

    red = int(p[0])
    green = int(p[1])
    blue = int(p[2])

    P = np.sqrt(red*red*Pr + green*green*Pg + blue*blue*Pb)

    red_v = truncate(P+(red-P)*value)
    green_v = truncate(P+(green-P)*value)
    blue_v = truncate(P+(blue-P)*value)

    return [red_v, green_v, blue_v]

# Adjusts the contrast on a pixel
def contrast(p, value):
    factor = (259 * (value + 255)) / float(255 * (259 - value))
    red = truncate(factor * (p[0] - 128) + 128)
    green = truncate(factor * (p[1] - 128) + 128)
    blue = truncate(factor * (p[2] - 128) + 128)
    return [red, green, blue]

#Increases the warmth or coolness of a pixel
def temperature(p,value):
    red = p[0]
    blue = p[2]
    if value > 0:
        red = truncate(p[0] + value)
    elif value < 0:
        blue = truncate(p[2] + value)

    return [red, p[1], blue]

#Inverts the colors, producing the same image that would be found in a film negative
def invert(p, value):
    if value:
        red = truncate(255 - p[0])
        green = truncate(255 - p[1])
        blue = truncate(255 - p[2])
    else:
        red = int(p[0])
        green = int(p[1])
        blue = int(p[2])

    return [red, green, blue]

#If the pixel is above value it becomes black, otherwise white
def threshold(p,value,apply):
    if apply:
        pixel_av = (p[0] + p[1] + p[2])/3.0

        if pixel_av>value:
            red = 255
            green = 255
            blue = 255
        else:
            red = 0
            green = 0
            blue = 0
    else:
        red = p[0]
        green = p[1]
        blue = p[2]

    return [red, green, blue]

# Ensures a pixel's value for a color is between 0 and 255
def truncate(value):
    if value < 0:
        value = 0
    elif value > 255:
        value = 255

    return value

# Applies the tilt-shift effect onto an image (grayscale for now)
# g_corner_x, and g_corner_y are needed in this Python
# implementation since we don't have thread methods to get our
# position.  Here they store the top left corner of the group.
# All of the work for a workgroup happens in one thread in
# this method
def tiltshift(input_image, output_image, buf,
              w, h,
              buf_w, buf_h, halo,
              l_w, l_h,
              bright, sat, con, temp, inv,
              g_corner_x, g_corner_y):

    # coordinates of the upper left corner of the buffer in image space, including halo
    buf_corner_x = g_corner_x - halo
    buf_corner_y = g_corner_y - halo

    # Load all pixels into the buffer from input_image
    # Loop over y values first, so we can load rows sequentially
    for row in range(0, buf_h):
        for col in range(0, buf_w):
            tmp_x = col
            tmp_y = row

            # Now ensure the pixel we are about to load is inside the image's boundaries
            if (buf_corner_x + tmp_x < 0) :
                tmp_x += 1
            elif (buf_corner_x + tmp_x >= w):
                tmp_x -= 1

            if (buf_corner_y + tmp_y < 0):
                tmp_y += 1
            elif (buf_corner_y + tmp_y >= h):
                tmp_y -= 1

            # Check you are within halo of global
            if ((buf_corner_y + tmp_y < h) and (buf_corner_x + tmp_x < w)):
                #input_image[((buf_corner_y + tmp_y) * w) + buf_corner_x + tmp_x];
                buf[row * buf_w + col] = input_image[buf_corner_y + tmp_y, buf_corner_x + tmp_x];
                
    # Loop over y first so we can calculate the blur amount
    for ly in range(0, l_h):
        # Initialize Global y Position
        y = ly + g_corner_y
        # Initialize Buffer y Position
        buf_y = ly + halo;

        for lx in range(0, l_w):
            # Initialize Global x Position
            x = lx + g_corner_x
            # Initialize Buffer x Position
            buf_x = lx + halo;

            # Stay in bounds check is necessary due to possible
            # images with size not nicely divisible by workgroup size
            if ((y < h) and (x < w)):

                p4 = buf[(buf_y * buf_w) + buf_x];
                p4 = brightness(p4,bright)
                p4 = saturation(p4, sat)
                p4 = temperature(p4,temp)
                p4 = invert(p4, inv)
                p4 = threshold(p4, thresh, apply_thresh)
                p4 = contrast(p4, con)

                output_image[y, x] = p4

    # Return the output of the last pass
    return output_image

# Rounds up the size to a be multiple of the group_size
def round_up(global_size, group_size):
    r = global_size % group_size
    if r == 0:
        return global_size
    return global_size + group_size - r

# generates horizontal blur mask using focus middle, focus radius, and image height,
# and stores the blur mask in the blur_mask parameter (np.array)
def generate_horizontal_blur_mask(blur_mask, middle_in_focus, in_focus_radius, height):
    # Calculate blur amount for each pixel based on middle_in_focus and in_focus_radius

    # Loop over y first so we can calculate the blur amount
    # fade out 20% to blurry so that there is not an abrupt transition
    no_blur_region = .8 * in_focus_radius
    # Set blur amount for focus middle
    #blur_row = np.array([blur_mask])
    blur_row = np.zeros_like(blur_mask[0], dtype=np.float)
    blur_mask[middle_in_focus] = blur_row
    # Simulataneously set blur amount for both rows of same distance from middle
    for y in xrange(middle_in_focus - in_focus_radius, middle_in_focus):
        # The blur amount depends on the y-value of the pixel
        distance_to_m = abs(y - middle_in_focus)
        # Note: Because we are iterating over all y's in the focus region, all the y's are within
        # the focus radius.
        # Thus, we need only check if we should fade to blurry so that there is not an abrupt transition
        if distance_to_m > no_blur_region:
            # Calculate blur ammount
            blur_amount = (1.0 / (in_focus_radius - no_blur_region)) * (distance_to_m - no_blur_region)
        else:
            # No blur
            blur_amount = 0.0
        blur_row.fill(blur_amount)
        if y > 0:
            blur_mask[y] = blur_row
        if middle_in_focus + distance_to_m < height:
            blur_mask[middle_in_focus + distance_to_m] = blur_row

# Generates a circular horizontal blur mask using the x and y coordinates of the focus middle,
# focus radius, and image height, and stores the blur mask in the blur_mask parameter (np.array)
def generate_circular_blur_mask(blur_mask, middle_in_focus_x, middle_in_focus_y, in_focus_radius, width, height):
    # Calculate blur amount for each pixel based on middle_in_focus and in_focus_radius

    # Fade out 20% to blurry so that there is not an abrupt transition
    no_blur_region = .8 * in_focus_radius
    # Set blur amount (no blur) for center of in-focus region
    #blur_mask[middle_in_focus_y, middle_in_focus_x] = 0.0

    # Calculate all x,y coords
    x_coords = np.arange(middle_in_focus_x - in_focus_radius, middle_in_focus_x + in_focus_radius + 1)
    y_coords = np.arange(middle_in_focus_y - in_focus_radius, middle_in_focus_y + +in_focus_radius + 1)
    xy_list = cartesian([x_coords, y_coords])
    size_arr = np.full(xy_list.shape,(width,height))
    # Filter out x,y coords that are out of bounds
    greater_than_zero = np.greater(xy_list,np.zeros(xy_list.shape))
    less_than_upper_bound = np.less(xy_list, size_arr)
    in_bounds_coords = np.logical_and(greater_than_zero,less_than_upper_bound)
    xy_list = xy_list[in_bounds_coords]
    #separate x's and y's into separate arrays
    xy_list = xy_list.T
    # Loop over x and y first so we can calculate the blur amount

    v_ciruclar_blur_mask_helper = np.vectorize(ciruclar_blur_mask_helper)
    xy_blur_amounts = v_ciruclar_blur_mask_helper(xy_list[0],xy_list[1], middle_in_focus_x, middle_in_focus_y, in_focus_radius, no_blur_region)
    blur_mask[xy_list[1],xy_list[0]] = xy_blur_amounts

def ciruclar_blur_mask_helper(x, y, middle_in_focus_x, middle_in_focus_y, in_focus_radius, no_blur_region):
    #print x,y
    x_distance_to_m = np.absolute(x - middle_in_focus_x)
    y_distance_to_m = np.absolute(y - middle_in_focus_y)
    distance_to_m = (x_distance_to_m ** 2 + y_distance_to_m ** 2) ** 0.5
    blur_amount = 1.0
    # Note: Not all values we iterate over are within the focus region, so we must check
    if distance_to_m < no_blur_region:
        # No blur
        blur_amount = 0.0
    # Check if we should fade to blurry so that there is not an abrupt transition
    elif distance_to_m < in_focus_radius:
        blur_amount = (1.0 / (in_focus_radius - no_blur_region)) * (distance_to_m - no_blur_region)

    return blur_amount


def cartesian(arrays, out=None):
    # This code was adapted from
    # http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """
    dtype = arrays[0].dtype
    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out



# Run a Python implementation of ImageFilters
if __name__ == '__main__':
    # Start the clock
    start_time = time.time()
    setup_time = time.time()

#==============================================================================
#     Setup for parsing Command Line Args
#==============================================================================
    parser = argparse.ArgumentParser(description='Image Effects (in pure Python)')
    parser.add_argument('-i','--input', help='Input image file name',required=True)
    parser.add_argument('-o','--output',help='Output image file name (required to save new image)', required=False)
    parser.add_argument('-n','--n_passes', help='Number of box blur passes',required=False)
    parser.add_argument('-b','--bright',help='Brightness Level (between -100.0 and 100.0)', required=False)
    parser.add_argument('-s','--sat',help='Saturation Level (between 0.0 and 2.0)', required=False)
    parser.add_argument('-c','--con',help='Contrast Level (between 0 and 50)', required=False)
    parser.add_argument('-t','--temp',help='Temperature Level (between -255 and 255)', required=False)
    parser.add_argument('-v','--inv',help='Invert Colors (0 -> Normal, 1 -> Inverted)', required=False)
    parser.add_argument('-z','--thresh',help='Threshold Level (between 0 and 255, default is None)', required=False)
    parser.add_argument('-w','--inverse',help='Invert Colors (0 -> Normal, 1 -> Inverted)', required=False)
    parser.add_argument('-x','--x_center',help='X coord of center of focus region', required=False)
    parser.add_argument('-y','--y_center',help='Y coord of center of focus region', required=False)
    parser.add_argument('-r','--radius',help='Radius of focus region', required=False)
    parser.add_argument('-m','--blur_mask',help='Blur mask file name', required=False)
    parser.add_argument('-f','--focus',help='Focus (0 -> In Focus, 1 -> Consistent Blur, 2 -> Circular Tilt Shift, 3 -> Horizontal Tilt Shift', required=False)

#==============================================================================
#     Parse Command Line Args
#==============================================================================
    args = parser.parse_args()

    # Load the image
    try:
        input_image = mpimg.imread(args.input,0)
    except (OSError, IOError) as e:
        parser.error('Valid input image file name required')

    width = np.int32(input_image.shape[1])
    height = np.int32(input_image.shape[0])

    # Output image file name
    out_filename = args.output if args.output is not None else None

    output_image = np.zeros_like(input_image)

   # Number of Passes - 3 passes approximates Gaussian Blur
    num_passes = np.int32(args.n_passes) if args.n_passes is not None else np.int32(3)
    if num_passes < 0:
        parser.error('Number of passes must be greater than 0')
    # Brightness - Between -100 and 100
    bright = np.float32(args.bright) if args.bright is not None else np.float32(0.0)
    if bright > 100 or bright < -100:
        parser.error('Brightness must be between -100 and 100')
    # Saturation - Between 0 and 2, 1.0 does not produce any effect
    sat = np.float32(args.sat) if args.sat is not None else np.float32(1.0)
    if sat > 5 or bright < 0:
        parser.error('Saturation must be between 0 and 5')
    # Contrast - Between -255 and 255
    con = np.float32(args.con) if args.con is not None else np.float32(0.0)
    if con > 255 or con < -255:
        parser.error('Contrast must be between -255 and 255')
    # Temperature - Between -255 and 255
    temp = np.int32(args.temp) if args.temp is not None else np.int32(0)
    if temp > 255 or temp < -255:
        parser.error('Temperature must be between -255 and 255')
    # Invert - True or False
    inv = np.bool_(True) if args.inverse == '1' else np.bool_(False)
    if inv is np.bool(False) and args.inverse not in ['1', None]:
        parser.error('Inverse must be 0 or 1')
    #Threshold - Between 0 and 255
    thresh = np.float32(-1.0)
    apply_thresh = False
    if args.thresh is not None:
        thresh = np.float32(args.thresh)
        apply_thresh = True
        if thresh > 255 or thresh < 0:
            parser.error('Threshold must be between 0 and 255')

    # Focus Type (default None)
    # Consistent blur
    consistent_blur = True if args.focus == '1' else False
    # Circle in-focus region
    focused_circle = True if args.focus == '2' else False
    # Horizontal in-focus region
    focused_hor = True if args.focus == '3' else False

    # The y-index of the center of the in-focus region
    middle_in_focus_y = np.int32(args.y_center) if args.y_center is not None else np.int32(height/2)
    if middle_in_focus_y > height or middle_in_focus_y < 0:
        parser.error('Y coord of center of focus region must be between 0 and {} for this image'.format(height-1))
    # The x-index of the center of the in-focus region
    # Note: this only matters for circular in-focus region
    middle_in_focus_x = np.int32(args.x_center) if args.x_center is not None else np.int32(width/2)
    if middle_in_focus_x > width or middle_in_focus_x < 0:
        parser.error('X coord of center of focus region must be between 0 and {} for this image'.format(width-1))
    # The number of pixels distance from middle_in_focus to keep in focus
    in_focus_radius = np.int32(args.radius) if args.radius is not None else np.int32(min(width, height)/2)
    if in_focus_radius < 0:
        parser.error('Radius of focus region must be positive')
    # Accept the file name storing the blur mask
    # Note: There is one float blur amount per pixel

    # If Tilt Shift is enabled
    if consistent_blur or focused_circle or focused_hor or args.blur_mask != None:
        # Initialize blur mask to be all 1's (completely blurry)
        # Note: There is one float blur amount per pixel
        if args.blur_mask is not None:
            blur_mask = mpimg.imread(args.blur_mask,0)
        else:
            # Initialize blur mask to be all 1's (completely blurry)
            blur_mask = np.ones(input_image.shape[:2], dtype=np.float32)
            # Generate the blur mask
            if focused_circle:
                print "Creating circular blur mask"
                generate_circular_blur_mask(blur_mask, middle_in_focus_x, middle_in_focus_y, in_focus_radius, width, height)
            elif focused_hor:
                print "Creating horizontal blur mask"
                generate_horizontal_blur_mask(blur_mask, middle_in_focus_y, in_focus_radius, height)
    else:
        # No blurring
        blur_mask = np.zeros(input_image.shape[:2], dtype=np.float32)
        num_passes = 1

    if blur_mask.shape != input_image.shape[:2]:
        parser.error('The specified blur mask\'s shape did not match the input image\'s shape')

#==============================================================================
#     End Parsing Command Line Args
#==============================================================================
    setup_end_time = time.time()
    print "Took {} seconds to setup".format(setup_end_time - setup_time)

    local_size = np.array([256, 256])  # This doesn't really affect speed for the Python implementation
    # We need to add [1:] because the first element in this list is the number of colors in RGB, namely 3
    gsizes = np.array(input_image.shape[::-1][1:])
    #get global dimensions using vectorized ops
    round_up_vector = np.vectorize(round_up)
    global_size = round_up_vector(gsizes, local_size)
    # Set up a (N+2 x N+2) local memory buffer.
    # +2 for 1-pixel halo on all sides
    #local_memory = [[]] * (local_size[0] + 2) * (local_size[1] + 2) #np.zeros((local_size[0] + 2) * (local_size[1] + 2))
    local_memory = [[]] * np.prod(local_size+2) #np.zeros((local_size[0] + 2) * (local_size[1] + 2))
    # Each work group will have its own private buffer.
    buf_width, buf_height = local_size + 2
    halo = np.int32(1)

    print "Image Width %s" % width
    print "Image Height %s" % height

    np_t = np.bool_(True)
    np_f = np.bool_(False)

    # Generate list of global corner coordinates to iterate over
    global_corner_x_coords = np.arange(0, global_size[0], local_size[0])
    global_corner_y_coords = np.arange(0, global_size[1], local_size[1])
    global_corner_xy_list = cartesian([global_corner_x_coords, global_corner_y_coords])
    # Loop over all groups and call tiltshift once per group
    for (group_corner_x, group_corner_y) in global_corner_xy_list:
        # Run tilt shift over the group and store the results in host_image_tilt_shifted
        tiltshift(input_image, output_image, local_memory,
                      width, height,
                      buf_width, buf_height, halo,
                      local_size[0], local_size[1],
                      bright, sat, con, temp, inv,
                      group_corner_x, group_corner_y)

    # Now put the output of the last pass into the input of the next pass
    input_image = output_image
    
    blur_time = time.time()
    print "Performing boxblur"
    
    input_image = numpy_boxblur(input_image, blur_mask, num_passes)

    end_time = time.time()
    print "TOTAL - Took %s seconds to run %s passes" % (end_time - start_time, num_passes)
    print "Blur time %s" % (end_time - blur_time)
    
    if out_filename is not None:
        # Save image
        mpimg.imsave(out_filename, input_image)
    else:
        # Display the new image
        plt.imshow(input_image)
        plt.show()