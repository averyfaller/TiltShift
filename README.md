# Image Filters with OpenCL

#### A library for applying filters to images.
1. Tilt Shift Effect (circular or horizontal focus region)
2. Saturation Adjustment
3. Contrast Adjustment
4. Brightness Adjustment
5. Threshold Filtering
6. Temperature Adjustment
7. Color Inversion
8. Blurring


In this project, we provide code to perform image manipulations, including Box Blur, as well as Saturation and Contrast adjustments.
Taken together, these effects can be combined to produce images which appear to be Tilt Shifted.

We provide code to perform these effects in both pure Python and OpenCL, which runs on GPUs.

## How to use the library (from the UNIX command line):
Note: For this example we will run the program OpenCL/ImageFilters.py to filter the NY.jpg image
### Name: 
    ./ImageFilters.py -- filters image
### Synopsis: 
    ./ImageFilters.py [-bcfilmnorstwxyz] [input_file ...]
### Description: 
Given an image file as input, the program modifies the image using a specified filter.

#### The following options are available:
##### -b , --bright
Brightness Level (between -100.0 and 100.0)
##### -c, --con
Contrast Level (between 0 and 50)
##### -f, --tilt_shift
Tilt Shift (0 -> Tilt Shift, 1 -> No Tilt Shift
##### -i, --input
 Input image file name
##### -l, --circle
Focus Region Shape (0 -> horizontal, 1 -> circle)
##### -m, --blur_mask
Blur mask file name
##### -n --n_passes
Number of box blur passes
##### -o, --output
Output image file name (required to save new image)
##### -r, --radius
Radius of focus region
##### -s, --sat
Saturation Level (between 0.0 and 2.0)
##### -t, --temp
Temperature Level (between -255 and 255)'
##### -w, --inverse
Invert Colors (0 -> Normal, 1 -> Inverted)
##### -x, --x_center
X coord of center of focus region
##### -y, --y_center
Y coord of center of focus region
##### -z, --thresh
Threshold Level (between 0 and 255, default is None)

### Example: Tilt Shift with circular focus region using 4 passes of box blur and with all other default settings
    ./ImageFilters.py -i ./Input/NY.jpg -l 1 -n 4
