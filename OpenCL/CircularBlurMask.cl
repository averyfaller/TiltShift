// Generates a circular blur mask
__kernel void 
blurmask(__global float* blur_mask, 
          int w, int h, 
          int middle_in_focus_x, int middle_in_focus_y, int in_focus_radius) {

    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);
        
    // Stay in bounds check is necessary due to possible 
    // images with size not nicely divisible by workgroup size
    if ((y < h) && (x < w)) {
        int x_distance_to_m = abs(x - middle_in_focus_x);
        int y_distance_to_m = abs(y - middle_in_focus_y);
        
        // The blur amount depends on the euclidean distance between the pixel and focus center
        float distance_to_m = sqrt((float)((x_distance_to_m * x_distance_to_m) + (y_distance_to_m * y_distance_to_m)));
        
        if (distance_to_m < in_focus_radius) {
            // No blur,
            float blur_amount = 0.0;
            
            // UNLESS...we're on the edges
            // Check if we should fade to blurry so that there is not an abrupt transition
            // Fade out 20% to blurry so that there is not an abrupt transition
            int no_blur_region = .8 * in_focus_radius;
            if (distance_to_m > no_blur_region) {
                blur_amount = (1.0 / (in_focus_radius - no_blur_region)) * (distance_to_m - no_blur_region);
            }
        
             blur_mask[y * w + x] = blur_amount;
        }
    }
}