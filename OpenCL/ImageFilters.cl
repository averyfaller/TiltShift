// Ensures a pixel's value for a color is between 0 and 255
inline uchar4 truncate(int4 p) {
    int red = p.y;
    int green = p.z;
    int blue = p.w;
    
    if (red < 0) {
        red = 0;
    } else if (red > 255) {
        red = 255;
    }
    
    if (green < 0) {
        green = 0;
    } else if (green > 255) {
        green = 255;
    }
    
    if (blue < 0) {
        blue = 0;
    } else if (blue > 255) {
        blue = 255;
    }
    
    uchar4 new_value = {p.x, red, green, blue};
    return new_value;
}

// Adjusts the brightness on a pixel    
inline int4 brightness(int4 p, float value) {
    int red = p.y + value;
    int green = p.z + value;
    int blue = p.w + value;

    int4 new_value = {p.x, red, green, blue};
    return new_value;
}

// Adjusts the saturation of a pixel    
inline int4 saturation(int4 p, float value) {
    float P = sqrt((p.y * p.y * .299) + (p.z * p.z * .587) + (p.w * p.w * .114));
        
    int red_v = P + ((p.y - P) * value);
    int green_v = P + ((p.z - P) * value);
    int blue_v = P + (((float)p.w - P) * value);

    int4 new_value = {p.x, red_v, green_v, blue_v};
    return new_value;
}
    
// Adjusts the contrast of a pixel    
inline int4 contrast(int4 p, float value) {
    float factor = (259 * (value + 255.0)) / (255 * (259.0 - value));
    int red_v = factor * (p.y - 128) + 128;
    int green_v = factor * (p.z - 128) + 128;
    int blue_v = factor * (p.w - 128) + 128;

    int4 new_value = {p.x, red_v, green_v, blue_v};
    return new_value;
}

// Increases the warmth or coolness of a pixel
inline int4 temperature(int4 p, float value) {
    int4 new_value = p;
    
    if (value > 0) {
        new_value.y = p.y + value;
    } else if (value < 0) {
        new_value.w = p.w - value;
    }

    return new_value;        
}            

// Inverts the colors, producing the same image that would be found in a film negative
inline int4 invert(int4 p, bool value) {
    if (value) {
        int red = 255 - p.y;
        int green = 255 - p.z;
        int blue = 255 - p.w;
        int4 new_value = {p.x, red, green, blue};
        return new_value;
    } else {
        return p;
    }
}

// If the pixel is above value it becomes black, otherwise white
inline int4 threshold(int4 p, float value) {
    int4 new_value = p;
    if (value > 0) {
        float pixel_avg = (p.y + p.z + p.w) / 3.0;
        
        if (pixel_avg > value) {
            new_value.y = 255;
            new_value.z = 255;
            new_value.w = 255;
        } else {
            new_value.y = 0;
            new_value.z = 0;
            new_value.w = 0;
        }
        return new_value;
    }
    return new_value;
}

// A method that takes in a matrix of 3x3 pixels and blurs 
// the center pixel based on the surrounding pixels, a 
// bluramount of 1 is full blur and will weight the neighboring
// pixels equally with the pixel that is being modified.  
// While a bluramount of 0 will result in no blurring.
inline uint boxblur(uchar4 p0, uchar4 p1, uchar4 p2, 
                    uchar4 p3, uchar4 p4, uchar4 p5, 
                    uchar4 p6, uchar4 p7, uchar4 p8) {
    
    float blur_amount = (float) p4.x / 255.0;
    // Calculate the blur amount for the central and 
    // neighboring pixels
    float self_blur_amount = (9 - (blur_amount * 8)) / 9;
    float other_blur_amount = (blur_amount / 9);
    
    // Sum a weighted average of self and others based on the blur amount
    uchar red_v = (self_blur_amount * p4.y) + (other_blur_amount * (p0.y + p1.y + p2.y + p3.y + p5.y + p6.y + p7.y + p8.y));
    uchar green_v = (self_blur_amount * p4.z) + (other_blur_amount * (p0.z + p1.z + p2.z + p3.z + p5.z + p6.z + p7.z + p8.z));
    uchar blue_v = (self_blur_amount * p4.w) + (other_blur_amount * (p0.w + p1.w + p2.w + p3.w + p5.w + p6.w + p7.w + p8.w));
    
    //uchar4 blur = {p4.x, red_v, green_v, blue_v};
    uint blur = (p4.x << 24) + (red_v << 16) + (green_v << 8) + blue_v;
    return blur;
}

// Applies the tilt-shift effect onto an image (grayscale for now)
// g_corner_x, and g_corner_y are needed in this Python 
// implementation since we don't have thread methods to get our 
// position.  Here they store the top left corner of the group.
// All of the work for a workgroup happens in one thread in 
// this method
__kernel void
tiltshift(__global __read_only uchar4* in_values, 
          __global __write_only uint* out_values, 
          __local uchar4* buf, 
          int w, int h, 
          int buf_w, int buf_h, 
          const int halo,
          float bright, float sat, float con, float temp, bool inv, float thresh,
          int pass_num) {

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
               
            uchar4 expanded = in_values[((buf_corner_y + tmp_y) * w) + buf_corner_x + tmp_x];

            // If we're in the first pass, perform the saturation, contrast, etc. adjustments
            if (pass_num == 0) {
                int4 int_expanded = {expanded.x, expanded.y, expanded.z, expanded.w};
                int_expanded = brightness(int_expanded, bright);
                int_expanded = saturation(int_expanded, sat);
                int_expanded = contrast(int_expanded, con);
                int_expanded = temperature(int_expanded, temp);
                int_expanded = invert(int_expanded, inv);
                int_expanded = threshold(int_expanded, thresh);
                expanded = truncate(int_expanded);
            }
            
            buf[row * buf_w + idx_1D] = expanded;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
        
    // Stay in bounds check is necessary due to possible 
    // images with size not nicely divisible by workgroup size
    if ((y < h) && (x < w)) {
        uchar4 p0 = buf[((buf_y - 1) * buf_w) + buf_x - 1];
        uchar4 p1 = buf[((buf_y - 1) * buf_w) + buf_x];
        uchar4 p2 = buf[((buf_y - 1) * buf_w) + buf_x + 1];
        uchar4 p3 = buf[(buf_y * buf_w) + buf_x - 1];
        uchar4 p4 = buf[(buf_y * buf_w) + buf_x];
        uchar4 p5 = buf[(buf_y * buf_w) + buf_x + 1];
        uchar4 p6 = buf[((buf_y + 1) * buf_w) + buf_x - 1];
        uchar4 p7 = buf[((buf_y + 1) * buf_w) + buf_x];
        uchar4 p8 = buf[((buf_y + 1) * buf_w) + buf_x + 1];
                
        // Perform boxblur
        uint blurred_pixel = boxblur(p0, p1, p2, p3, p4, p5, p6, p7, p8);
        out_values[y * w + x] = blurred_pixel;
    }
}