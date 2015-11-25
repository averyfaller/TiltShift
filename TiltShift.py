# A basic, parallelized Python implementation of 
# the Tilt-Shift effect we hope to achieve in OpenCL

# A method that takes in a matrix of 3x3 pixels and blurs 
# the center pixel based on the surrounding pixels, a 
# bluramount of 1 is full blur and will weight the neighboring
# pixels equally with the pixel that is being modified.  
# While a bluramount of 0 will result in no blurring.
def boxblur(pixelmatrix, bluramount):
    # Calculate the blur amount for the central and 
    # neighboring pixels
    selfbluramount = (9 - bluramount * 8) / 9.0
    otherbluramount = bluramount / 9.0
    # TODO
    

# Applies the tilt-shift effect onto an image
def tiltshift(image, num_passes=3):
    # We will perform 3 passes of the bux blur 
    # effect to approximate Gaussian blurring
    
    local_size = (8, 8)  # 64 pixels per work group
    global_size = tuple([round_up(g, l) for g, l in zip(host_image.shape[::-1], local_size)])
    width = np.int32(host_image.shape[1])
    height = np.int32(host_image.shape[0])
    
    # Set up a (N+2 x N+2) local memory buffer.
    # +2 for 1-pixel halo on all sides
    local_memory = np.zeroes((local_size[0] + 2), (local_size[1] + 2))
    
    # Each work group will have its own private buffer.
    buf_width = np.int32(local_size[0] + 2)
    buf_height = np.int32(local_size[1] + 2)
    halo = np.int32(1)
    
    for p in range(num_passes):
        # Loop over y first so we can calculate the bluramount
        for y:
            bluramount = 1
            for x:
                # TODO
                 = boxblur(pixelmatrix, bluramount)
                
    
# Adjusts the saturation of an image    
def saturation():
    # TODO
    
# Adjusts the contrast on an image    
def contrast():
    # TODO
    

# Run a Python implementation of Tilt-Shift
if __name__ == '__main__':
    host_image = np.load('image.npz')['image'].astype(np.float32)[::2, ::2].copy()
    host_image_tilt_shifted = np.zeros_like(host_image)

    num_passes = 3
    
    tiltshift(host_image, num_passes)