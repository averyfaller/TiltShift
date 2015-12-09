# Image Filters with OpenCL
Sam Daulton, Avery Faller, Isadora Nun

A Final Project for CS205

#### A library for applying filters to images.
1. Tilt Shift Effect (circular or horizontal focus region, or consistent blur)
2. Saturation Adjustment
3. Contrast Adjustment
4. Brightness Adjustment
5. Threshold Filtering
6. Temperature Adjustment
7. Color Inversion
8. Blurring


In this project, we provide code to perform image manipulations, including Box Blur, as well as Saturation and Contrast adjustments.
Taken together, these effects can be combined to produce images which appear to be Tilt Shifted.

We provide code to perform these effects in both pure Python (Python/) and OpenCL (OpenCL/), which can be run on computers with GPUs.  In the OpenCL folder, you will find two implementations, a baseline implementation (OpenCL/ImageFilters.py) and an optimized implementation (OpenCL/ImageFilters.py).

## How to use the library (from the UNIX command line):
Note: For this example we will run the program OpenCL/ImageFilters.py to filter the NY.jpg image
### Basic: 
    python ./ImageFilters.py -i [path_to_input_file] 
This will run the image at the specified path through the ImageFilters code.  Add effects by specifying which effect you want to run and the amount.  For a list of all available effects run:
    python ImageFilters.py -h

### Description: 
Given an image file as input, the program modifies the image using specified filters.

#### Required parameters:
##### -i, --input
Input image file name

#### General optional parameters:
##### -b , --bright
Brightness Level (between -100.0 and 100.0)
##### -c, --con
Contrast Level (between -255 and 255)
##### -o, --output
Output image file name (required to save new image)
##### -s, --sat
Saturation Level (between 0.0 and 5.0)
##### -t, --temp
Temperature Level (between -255 and 255)'
##### -w, --inverse
Invert Colors (0 -> Normal, 1 -> Inverted)
##### -z, --thresh
Threshold Level (between 0 and 255, default is None)

#### Focus-specific parameters:
##### -f, --focus
Focus (0 -> In Focus, 1 -> Consistent Blur, 2 -> Circular Tilt Shift, 3 -> Horizontal Tilt Shift)
##### -m, --blur_mask
Blur mask file name
##### -n, --n_passes
Number of box blur passes
##### -r, --radius
Radius of focus region
##### -x, --x_center
X coord of center of focus region
##### -y, --y_center
Y coord of center of focus region

### Examples:
#### Circular Tilt Shift using 4 passes of box blur, increased contrast, saturation and warmth:
    python ./ImageFilters.py -i ../Input/NY.jpg -c 20 -s 2.0 -t 30 -f 2 -n 4 -r 50 -x 350 -y 430
#### Increased contrast, saturation and warmth, no blurring:
    python ./ImageFilters.py -i ../Input/NY.jpg -c 20 -s 2.0 -t 50
#### Increased contrast and apply a cutoff threshold of 70:
    python ./ImageFilters.py -i ../Input/NY.jpg -c 20 -z 70
