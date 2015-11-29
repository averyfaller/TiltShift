// A method that takes in a matrix of 3x3 pixels and blurs 
// the center pixel based on the surrounding pixels, a 
// bluramount of 1 is full blur and will weight the neighboring
// pixels equally with the pixel that is being modified.  
// While a bluramount of 0 will result in no blurring.
inline uint3 boxblur(float blur_amount, 
                     uint3 p0, uint3 p1, uint3 p2, 
                     uint3 p3, uint3 p4, uint3 p5, 
                     uint3 p6, uint3 p7, uint3 p8) {
    
    // Calculate the blur amount for the central and 
    // neighboring pixels
    float self_blur_amount = (9 - (blur_amount * 8)) / 9.0;
    float other_blur_amount = blur_amount / 9.0;
    
    // Sum a weighted average of self and others based on the blur amount
    // print "--------"
    // print 0 + p0[0] + p1[0] + p2[0] + p3[0] + p5[0] + p6[0] + p7[0] + p8[0]

    uint red_v = (self_blur_amount * p4.x) + (other_blur_amount * (0 + p0.x + p1.x + p2.x + p3.x + p5.x + p6.x + p7.x + p8.x));
    uint blue_v = (self_blur_amount * p4.y) + (other_blur_amount * (0 + p0.y + p1.y + p2.y + p3.y + p5.y + p6.y + p7.y + p8.y));
    uint green_v = (self_blur_amount * p4.z) + (other_blur_amount * (0 + p0.z + p1.z + p2.z + p3.z + p5.z + p6.z + p7.z + p8.z));
        
    uint3 blur = {red_v, blue_v, green_v};
    // print "Original %s" % p4
    // print "New %s" % [int(red_v), int(blue_v), int(green_v)]
    // print "--------"
    return blur;
}

// Applies the tilt-shift effect onto an image (grayscale for now)
// g_corner_x, and g_corner_y are needed in this Python 
// implementation since we don't have thread methods to get our 
// position.  Here they store the top left corner of the group.
// All of the work for a workgroup happens in one thread in 
// this method
__kernel void
tiltshift(__global __read_only uint3* in_values, 
          __global __write_only uint3* out_values, 
          __local uint3* buf, 
          int w, int h, 
          int buf_w, int buf_h, 
          const int halo,
          float sat, float con, bool last_pass,
          int focus_m, int focus_r) {

    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;
        
    int row;

    // Since the kernels are in order, by loading by column per kernel, 
    // we'll actually be loading by row across kernels
    if (idx_1D < buf_w) {
        for (row = 0; row < buf_h; row++) {
            int tmp_x = idx_1D;
            int tmp_y = row;
            
            if (buf_corner_x + tmp_x < 0) {
                tmp_x++;
            } else if (buf_corner_x + tmp_x >= w) {
                tmp_x--;
            }
            
            if (buf_corner_y + tmp_y < 0) {
                tmp_y++;
            } else if (buf_corner_y + tmp_y >= h) {
                tmp_y--;
            }
                
            buf[row * buf_w + idx_1D] = in_values[((buf_corner_y + tmp_y) * w) + buf_corner_x + tmp_x];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    // The blur amount depends on the y-value of the pixel
    float blur_amount = 1.0;
    int distance_to_m = abs(y - focus_m);

    // The edge of the in-focus area should fade to blurry so that there is not an abrupt transition
    float no_blur_region = .8 * focus_r;
    // If it is within the middle 90% then don't have any blur at all, but then linearly increase to 1.0
    if (distance_to_m < no_blur_region) {
        blur_amount = 0;
    } else if (distance_to_m < focus_r) {
        blur_amount = (1.0 / (focus_r - no_blur_region)) * (distance_to_m - no_blur_region);
    }

    // Stay in bounds check is necessary due to possible 
    // images with size not nicely divisible by workgroup size
    if ((y < h) && (x < w)) {
        uint3 p0 = buf[((buf_y - 1) * buf_w) + buf_x - 1];
        uint3 p1 = buf[((buf_y - 1) * buf_w) + buf_x];
        uint3 p2 = buf[((buf_y - 1) * buf_w) + buf_x + 1];
        uint3 p3 = buf[(buf_y * buf_w) + buf_x - 1];
        uint3 p4 = buf[(buf_y * buf_w) + buf_x];
        uint3 p5 = buf[(buf_y * buf_w) + buf_x + 1];
        uint3 p6 = buf[((buf_y + 1) * buf_w) + buf_x - 1];
        uint3 p7 = buf[((buf_y + 1) * buf_w) + buf_x];
        uint3 p8 = buf[((buf_y + 1) * buf_w) + buf_x + 1];
                
        // Perform boxblur
        uint3 blurred_pixel = boxblur(blur_amount, p0, p1, p2, p3, p4, p5, p6, p7, p8);
                    
        // If we're in the last pass, perform the saturation and contrast adjustments as well
        if (last_pass) {
        //    blurred_pixel = saturation(blurred_pixel, sat);
        //    blurred_pixel = contrast(blurred_pixel, con);
        }
        out_values[y * w + x] = blurred_pixel;
    }
}
