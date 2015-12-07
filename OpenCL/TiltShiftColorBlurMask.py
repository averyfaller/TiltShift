import pyopencl as cl
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import color
from cython.parallel import prange
import time
import math

# A basic, parallelized Python implementation of 
# the Tilt-Shift effect we hope to achieve in OpenCL

# Rounds up the size to a be multiple of the group_size
def round_up(global_size, group_size):
    r = global_size % group_size
    if r == 0:
        return global_size
    return global_size + group_size - r

# Generates a horizontal blur mask using focus middle, focus radius, and image height
# and stores the blur mask in the blur_mask parameter (np.array)
def generate_horizontal_blur_mask(blur_mask, middle_in_focus, in_focus_radius, height):
    # Calculate blur amount for each pixel based on middle_in_focus and in_focus_radius

    # Linearly fade out the last 20% on each side to blurry,
    # so that there is not an abrupt transition
    no_blur_region = .8 * in_focus_radius
    
    # Set blur amount for focus middle
    #blur_row = np.array([blur_mask])
    blur_row = np.zeros_like(blur_mask[0], dtype=np.float)
    blur_mask[middle_in_focus] = blur_row
    # Simulataneously set blur amount for both rows of same distance from middle
    # Loop over y first so we can calculate the blur amount
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
    
    # Loop over x and y first so we can calculate the blur amount
    for x in xrange(middle_in_focus_x - in_focus_radius, middle_in_focus_x + 1):
        for y in xrange(middle_in_focus_y - in_focus_radius, middle_in_focus_y + 1):
            
            # The blur amount depends on the euclidean distance between the pixel and focus center
            x_distance_to_m = abs(x - middle_in_focus_x)
            y_distance_to_m = abs(y - middle_in_focus_y)
            distance_to_m = (x_distance_to_m ** 2 + y_distance_to_m ** 2) ** 0.5
            
            blur_amount = 1.0
            # Note: Not all values we iterate over are within the focus region, so we must check
            if distance_to_m < no_blur_region:
                # No blur
                blur_amount = 0.0
            # Check if we should fade to blurry so that there is not an abrupt transition
            elif distance_to_m < in_focus_radius:
                blur_amount = (1.0 / (in_focus_radius - no_blur_region)) * (distance_to_m - no_blur_region)
            
            # Set the blur_amount for 4 pixels of the same distance from the center of the in-focus region
            if x > 0:
                if y > 0:
                    blur_mask[y, x] = blur_amount
                if middle_in_focus_y + y_distance_to_m < height:
                    blur_mask[middle_in_focus_y + y_distance_to_m, x] = blur_amount
            if middle_in_focus_x + x_distance_to_m < width:
                if y > 0:
                    blur_mask[y, middle_in_focus_x + x_distance_to_m] = blur_amount
                if middle_in_focus_y + y_distance_to_m < height:
                    blur_mask[middle_in_focus_y + y_distance_to_m, middle_in_focus_x + x_distance_to_m] = blur_amount

# Run a Python implementation of Tilt-Shift (grayscale)
if __name__ == '__main__':
    # Load the image
    input_image = mpimg.imread('../NY.JPG',0)
    
    # Start the clock
    start_time = time.time()
    
    conversion_start_time = time.time()
    # Convert the image from (h, w, 3) to (h, w), storing the RGB value into an int
    image_combined = (input_image[...,0].astype(np.uint32) << 16) + (input_image[...,1].astype(np.uint32) << 8) + (input_image[...,2].astype(np.uint32) << 0)
    conversion_end_time = time.time()
    print image_combined.shape
    
    # Make the placeholders for the output image and output combined
    #output_combined = np.zeros_like(image_combined)
    host_image_filtered = np.zeros_like(input_image)
        
    # List our platforms
    platforms = cl.get_platforms()
    print 'The platforms detected are:'
    print '---------------------------'
    for platform in platforms:
        print platform.name, platform.vendor, 'version:', platform.version
    
    # List devices in each platform
    for platform in platforms:
        print 'The devices detected on platform', platform.name, 'are:'
        print '---------------------------'
        for device in platform.get_devices():
            print device.name, '[Type:', cl.device_type.to_string(device.type), ']'
            print 'Maximum clock Frequency:', device.max_clock_frequency, 'MHz'
            print 'Maximum allocable memory size:', int(device.max_mem_alloc_size / 1e6), 'MB'
            print 'Maximum work group size', device.max_work_group_size
            print '---------------------------'

    # Create a context with all the devices
    devices = platforms[0].get_devices()
    context = cl.Context(devices[1:])
    print 'This context is associated with ', len(context.devices), 'devices'
    
    # Create a queue for transferring data and launching computations.
    # Turn on profiling to allow us to check event times.
    queue = cl.CommandQueue(context, context.devices[0],
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    print 'The queue is using the device:', queue.device.name

    curdir = os.path.dirname(os.path.realpath(__file__))
    program = cl.Program(context, open('TiltShiftColorBlurMask.cl').read()).build(options=['-I', curdir])
        
    buf_start_time = time.time()
    gpu_image_a = cl.Buffer(context, cl.mem_flags.READ_WRITE, image_combined.size * 32)
    gpu_image_b = cl.Buffer(context, cl.mem_flags.READ_WRITE, image_combined.size * 32)
    buf_end_time = time.time()
    
    # These settings for local_size appear to work best on my computer, not entirely sure why
    # (HD Graphics 4000 [Type: GPU] Maximum work group size 512)
    local_size = (256, 2)
    global_size = tuple([round_up(g, l) for g, l in zip(image_combined.shape[::-1], local_size)])

    width = np.int32(image_combined.shape[1])
    height = np.int32(image_combined.shape[0])
    
    # Set up a (N+2 x N+2) local memory buffer.
    # +2 for 1-pixel halo on all sides, 4 bytes for float.
    local_memory = cl.LocalMemory(4 * (local_size[0] + 2) * (local_size[1] + 2))
    # Each work group will have its own private buffer.
    buf_width = np.int32(local_size[0] + 2)
    buf_height = np.int32(local_size[1] + 2)
    halo = np.int32(1)
    
    ################################
    ### USER CHANGEABLE SETTINGS ###
    ################################
    ### General settings ###
    # Number of Passes - 3 passes approximates Gaussian Blur
    num_passes = 5
    # Brightness - Between -100 and 100
    bright = np.float32(10.0)
    # Saturation - Between 0.0 and 5.0
    sat = np.float32(2.0)
    # Contrast - Between -255 and 255
    con = np.float32(50.0)
    # Temperature - Between -255 and 255
    temp = np.float32(-30.0)
    # Invert - True or False
    inv = np.bool_(False)

    
    
    #### Tilt-Shift Settings ####
    # The y-index of the center of the in-focus region
    middle_in_focus_y = np.int32(280)
    # The number of pixels to either side of the middle_in_focus to keep in focus
    in_focus_radius = np.int32(200)

    # Circle specific settings
    # Circle in-focus region, or horizontal in-focus region
    focused_circle = True
    # The x-index of the center of the in-focus region
    middle_in_focus_x = np.int32(370)
    ####################################
    ### END USER CHANGEABLE SETTINGS ###
    ####################################
        
    # Initialize blur mask to be all 1's (completely blurry)
    # Note: There is one float blur amount per pixel
    blur_mask = np.ones_like(image_combined, dtype=np.float32)
    # Generate the blur mask
    if focused_circle:
        print "Genearting a circular blur mask around (%s, %s)" % (middle_in_focus_x, middle_in_focus_y)
        # Circular Blur Mask
        generate_circular_blur_mask(blur_mask, middle_in_focus_x, middle_in_focus_y, in_focus_radius, width, height)
    else:
        print "Genearting a horizontal blur mask %s" % middle_in_focus_y
        # Horizontal Blur Mask
        generate_horizontal_blur_mask(blur_mask, middle_in_focus_y, in_focus_radius, height)
    
    # Set the 4th parameter in the input image to be the blur amount for that pixel
    image_combined += ((255 * blur_mask).astype(np.uint32) << 24)

    # Send image to the device, non-blocking
    # This needs to be run after we update the image combined with our new values
    enqueue_start_time = time.time()
    cl.enqueue_copy(queue, gpu_image_a, image_combined, is_blocking=False)
    enqueue_end_time = time.time()
    
    print "Image Width %s" % width
    print "Image Height %s" % height
        
    kernel_start_time = time.time()
    # We will perform 3 passes of the bux blur 
    # effect to approximate Gaussian blurring
    for pass_num in range(num_passes):
        print "In iteration %s of %s" % (pass_num + 1, num_passes)
        # We need to loop over the workgroups here, 
        # because unlike OpenCL, they are not 
        # automatically set up by Python
        a_pass_num = np.int32(pass_num)
        if pass_num == 0:
            print "First Pass!"
            
        # Run tilt shift over the group and store the results in host_image_tilt_shifted
        # Loop over all groups and call tiltshift once per group    
        program.tiltshift(queue, global_size, local_size,
                          gpu_image_a, gpu_image_b, local_memory, 
                          width, height, 
                          buf_width, buf_height, halo,
                          bright, sat, con, temp, inv, 
                          a_pass_num)

        # Now put the output of the last pass into the input of the next pass
        gpu_image_a, gpu_image_b = gpu_image_b, gpu_image_a
    kernel_end_time = time.time()
    
    dequeue_start_time = time.time()
    cl.enqueue_copy(queue, image_combined, gpu_image_a, is_blocking=True)
    dequeue_end_time = time.time()
    
    reconversion_start_time = time.time()    
    host_image_filtered[...,0] = ((image_combined >> 16) & 0xFF)
    host_image_filtered[...,1] = ((image_combined >> 8) & 0xFF)
    host_image_filtered[...,2] = ((image_combined) & 0xFF)
    reconversion_end_time = time.time()
    
    end_time = time.time()
    print "####### TIMING BREAKDOWN #######"
    print "Took %s total seconds to run %s passes" % (end_time - start_time, num_passes)  
    print "Conversion time was %s seconds" % (conversion_end_time - conversion_start_time) 
    print "Buf creation time was %s seconds" % (buf_end_time - buf_start_time)
    print "Enqueue time was %s seconds" % (enqueue_end_time - enqueue_start_time) 
    print "Kernel time was %s seconds" % (kernel_end_time - kernel_start_time)  
    print "Dequeue time was %s seconds" % (dequeue_end_time - dequeue_start_time) 
    print "Reconversion time was %s seconds" % (reconversion_end_time - reconversion_start_time) 
    
    # Display the new image
    plt.imshow(host_image_filtered)    
    plt.show()
    mpimg.imsave("NY_FifthPass.png", host_image_filtered)