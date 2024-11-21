import cv2
import numpy as np

def bilateral_filter(image, d, sigma_color, sigma_space):
    result = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            i_min = max(0, i - d)
            i_max = min(image.shape[0], i + d + 1)
            j_min = max(0, j - d)
            j_max = min(image.shape[1], j + d + 1)

            region = image[i_min:i_max, j_min:j_max]

            intensity_diff = np.linalg.norm(region - image[i, j], axis=2)
            spatial_diff_i, spatial_diff_j = np.meshgrid(
                np.arange(i_min, i_max) - i,
                np.arange(j_min, j_max) - j,
                indexing='ij'
            )
            spatial_diff = np.sqrt(spatial_diff_i**2 + spatial_diff_j**2)

            weight = (
                np.exp(-intensity_diff**2 / (2 * sigma_color**2)) *
                np.exp(-spatial_diff**2 / (2 * sigma_space**2))
            )

            normalized_weight = weight / np.sum(weight)

            result[i, j] = np.sum(region * normalized_weight[:, :, None], axis=(0, 1))

    return result.astype(np.uint8)

def combine_flash_nonflash(flash_image, nonflash_image, d_noflash, sigma_color_noflash, sigma_space_noflash):
    nonflash_filtered = bilateral_filter(nonflash_image, d_noflash, sigma_color_noflash, sigma_space_noflash)
    combined_image = cv2.addWeighted(flash_image, 0.6, nonflash_filtered, 0.4, 0)
    return combined_image

def solution(image_path_a, image_path_b):
    ############################
    ############################
    ## image_path_a is path to the non-flash high ISO image
    ## image_path_b is path to the flash low ISO image
    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    flash_image = cv2.imread(image_path_b)
    nonflash_image = cv2.imread(image_path_a)

    d_flash = 20
    sigma_color_flash = 40
    sigma_space_flash = 40

    result_image = combine_flash_nonflash(flash_image, nonflash_image, d_flash, sigma_color_flash, sigma_space_flash)
    
    return result_image