import sys
import os
import numpy as np
import time

script_dir = os.path.dirname(__file__)
mymodule_dir1 = os.path.join(script_dir, '..', 'objects')
mymodule_dir2 = os.path.join(script_dir, '..', 'operators')
mymodule_dir3 = os.path.join(script_dir, '..', 'utils')
sys.path.append(mymodule_dir1)
sys.path.append(mymodule_dir2)
sys.path.append(mymodule_dir3)

from image import Image
from image_collection import ImageCollection
from reduce import reduce_image_collection
from spatial_utils import load_raster
import cupy as cp

# def check_size(data):
#     memory_size_bytes = data.nbytes

#     # Convert to kilobytes
#     memory_size_kb = memory_size_bytes / (1024*1024)

#     print(f"Size in memory: {memory_size_kb:.2f} KB")


# img1 = Image('../data/sr1_INDIA_2017_18.tif')
# img2 = Image('../data/sr2_INDIA_2017_18.tif')
# img3 = Image('../data/sr3_INDIA_2017_18.tif')

# rain1 = Image('../data/rain1_INDIA_2017_18_f.tif')
# rain2 = Image('../data/rain2_INDIA_2017_18_f.tif')
# rain3 = Image('../data/rain3_INDIA_2017_18_f.tif')
# rain4 = Image('../data/rain4_INDIA_2017_18_f.tif')
# rain5 = Image('../data/rain5_INDIA_2017_18_f.tif')

# rains1 = ImageCollection([rain1, rain2, rain3, rain4, rain5])
# rains2 = ImageCollection([rain3, rain4, rain5])
# rains3 = ImageCollection([img1, img2, img3])

# reduce_image_collection = rains1.reduce('mean')

# print(reduce_image_collection.check_dimensions())

# rains3_mean = rains3.reduce('mean')
# print(rains3_mean.check_dimensions())
# print(rains3.get_size())

# sr1 = img1.select('sr1')

# Assuming sr2 and antecedent are Image objects already loaded

# sr1_sample = sr1.get_sample(size=(3, 3), position='center')

# print(sr1_sample.check_dimensions())
# print(sr1_sample.get_data())
# start_time = time.time()
# M2 = Image.gpu_expression(
#     '0.5*(-sr+sqrt(sr**2+4*p*sr))', {
#         'sr': sr1.select('sr1'),
#         'p': sr1.select('sr1')
#     }
# ).rename('m2')
# end_time = time.time()

# execution_time = end_time - start_time
# print(f"Execution time: {execution_time} seconds")

# m2_sample = M2.get_sample(size=(3, 3), position='center')

# print(M2.check_dimensions())
# print(M2.get_data())


# runoff = Image.gpu_expression(
#     'np.where(np.logical_and.reduce([(P>=0.2*sr1), (P5>=0), (P5<=35), (((P-0.2*sr1)*(P-0.2*sr1+m1))/(P+0.2*sr1+sr1+m1))>=0]), ((P-0.2*sr1)*(P-0.2*sr1+m1))/(P+0.2*sr1+sr1+m1), ' +
#     'np.where(np.logical_and.reduce([(P>=0.2*sr2), (P5>=0), (P5>35), (((P-0.2*sr2)*(P-0.2*sr2+m2))/(P+0.2*sr2+sr2+m2))>=0]), ((P-0.2*sr2)*(P-0.2*sr2+m2))/(P+0.2*sr2+sr2+m2), ' +
#     'np.where(np.logical_and.reduce([(P>=0.2*sr3), (P5>=0), (P5>52.5), (((P-0.2*sr3)*(P-0.2*sr3+m3))/(P+0.2*sr3+sr3+m3))>=0]), ((P-0.2*sr3)*(P-0.2*sr3+m3))/(P+0.2*sr3+sr3+m3), ' +
#     '0)))',
#     {
#         'P': sr1.select('sr1'),
#         'm1': sr1.select('sr1'),
#         'm2': sr1.select('sr1'),
#         'm3': sr1.select('sr1'),
#         'P5': sr1.select('sr1'),
#         'sr2': sr1.select('sr1'),
#         'sr1': sr1.select('sr1'),
#         'sr3': sr1.select('sr1')
#     }
# ).rename('runoff')

# print(runoff.check_dimensions())
# print(runoff.get_data())


## starting ..............................

img1 = Image('../data/sr1_INDIA_2017_18.tif') # sr1
img2 = Image('../data/sr2_INDIA_2017_18.tif') # sr2
img3 = Image('../data/sr3_INDIA_2017_18.tif') # sr3

sr1 = cp.asarray(img1.get_data())
sr2 = cp.asarray(img2.get_data())
sr3 = cp.asarray(img3.get_data())

# custom_kernel = cp.ElementwiseKernel(
#     'float64 sr, float64 p',
#     'float64 z',
#     'z = 0.5f * (-sr + sqrt(sr*sr + 4*p*sr))',
#     'custom_operation'
# )

# start_time = time.time()
# M2 = custom_kernel(sr1, sr1)
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Execution time: {execution_time} seconds")


# reducer kernel .........................

sum_kernel = cp.ElementwiseKernel(
    'float64 x1, float64 x2, float64 x3, float64 x4, float64 x5',
    'float64 y',
    'y = x1 + x2 + x3 + x4 + x5',
    'sum_operation'
)

def sum_images(images):
    assert all(img.shape == images[0].shape for img in images), "All images must have the same shape"
    
    num_bands = images[0].shape[0]
    result = cp.zeros_like(images[0])
    
    for band in range(num_bands):
        band_data = [img[band] for img in images]
        result[band] = sum_kernel(*band_data)
    
    return result


start_time = time.time()
rain_sum = sum_images([sr1, sr2, sr3, sr1, sr1])
end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")



# reprojection kernel ......................................


# def gpu_reproject(input_image, input_transform, output_transform, output_shape, resampling_method='nearest'):
#     # Load image data to GPU
#     gpu_image = cp.asarray(input_image)

#     # Define CUDA kernel for reprojection
#     reproject_kernel = cp.ElementwiseKernel(
#         'raw T input, raw T input_transform, raw T output_transform',
#         'T output',
#         '''
#         // Calculate input coordinates from output coordinates
#         float x = i % output_width;
#         float y = i / output_width;
#         float lon = output_transform[0] + x * output_transform[1] + y * output_transform[2];
#         float lat = output_transform[3] + x * output_transform[4] + y * output_transform[5];
        
#         // Transform to input image coordinates
#         float col = (lon - input_transform[0]) / input_transform[1];
#         float row = (lat - input_transform[3]) / input_transform[5];
        
#         // Perform resampling
#         if (resampling_method == 'nearest') {
#             int nearest_col = round(col);
#             int nearest_row = round(row);
#             if (nearest_col >= 0 && nearest_col < input_width && 
#                 nearest_row >= 0 && nearest_row < input_height) {
#                 output = input[nearest_row * input_width + nearest_col];
#             }
#         }
#         // Add other resampling methods (bilinear, cubic) here
#         ''',
#         'reproject'
#     )

#     # Execute kernel
#     output = cp.zeros(output_shape, dtype=gpu_image.dtype)
#     reproject_kernel(gpu_image, input_transform, output_transform, output)

#     return cp.asnumpy(output)


# output_shape = (6365, 8358)





























