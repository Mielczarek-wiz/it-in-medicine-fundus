from tools import *

#funkcja zwraca binarną maskę
def image_processing(input_bitmap):
    initial_result = initial_processing(input_bitmap)
    
    advanced_result = advanced_processing(initial_result, input_bitmap)

    final_result = final_processing(advanced_result)
    
    return final_result
	
#funkcja przeprowadza wstępne przetwarzanie obrazu
def initial_processing(input_bitmap):
    sigma = 3.0
    
    result = exposure.equalize_hist(input_bitmap)
    result = gaussian(result, sigma=(sigma, sigma), truncate=3.5, multichannel=True)
    result = unsharp_mask(result, radius=3, amount=1)
    result = rgb2gray(result)
    
    MIN = np.percentile(result, 0.0)
    MAX = np.percentile(result, 100.0)
    result = (result - MIN) / (MAX - MIN)

    return result
	

#funkcja przeprowadza główną część przetwarzania obrazu
def advanced_processing(initial_result, input_bitmap):
    white_eye = rgb2gray(input_bitmap.copy())
    white_eye[white_eye > 0.02] = 1.0
    white_eye = sobel(white_eye)
    white_eye = mp.dilation(white_eye, mp.disk(5))
    white_eye = mp.dilation(white_eye, mp.disk(5))
    
    result = sobel(initial_result)
    result[result[:,:] < 0.02] = 0
    result[result[:,:] > 0] = 1
    
    result = mp.dilation(result, mp.disk(3))

    for y in range(len(result)):
        for x in range(len(result[y])):
            if(white_eye[y][x] > 0.5):
                result[y][x] = 0.0
    
    return result
	
#funkcja przeprowadza końcowe poprawki i zwraca binarną maskę
def final_processing(advanced_result):
    result = mp.dilation(advanced_result, mp.disk(5))
    
    result = mp.erosion(result, mp.disk(5))
    
    return result
	
def simple(path_test='test/image/*.jpg', path_label='test/label/*.tif', how_many=2):
    test_bitmaps = io.ImageCollection(path_test, as_gray=False)
    expert_masks = io.ImageCollection(path_label, as_gray=True)
    
    for i in range(how_many):
        plt.figure()
        io.imshow(test_bitmaps[i]) #, cmap='gray'
        plt.figure()
        io.imshow(expert_masks[i], cmap='gray') #, cmap='gray'
        binary_mask = image_processing(test_bitmaps[i])
        #print(binary_mask)
        plt.figure()
        io.imshow(binary_mask, cmap='gray')
        statistical_analysis(binary_mask, expert_masks[i])

	