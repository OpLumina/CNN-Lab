import numpy as np
random_array = np.random.rand(28, 28)

def relu(number):
    number = n
    if n <= 0:
        n = 0
        return n
    else:
        return n


# Extracts an array from another array, x and y coordinate starting from top left, width = width of array to extract, height = height of array to extract, array = array to extract from
def extract_array_from_array(x,y,width,height,array):
    extracted_array = array[y:y+height, x:x+width].copy()
    return extracted_array



"""
Note: array.shape returns height, width not width,height
Example:
random_array = np.random.rand(4, 5)
y,x = random_array.shape
print(x,y)
print(random_array)
"""

def feature_map(filter_array, main_array, stride, bias)
    # Calculate window array's height and width based on the size of the filter
    window_height, window_width = filter_array.shape

    # Find Height and Width of main array
    main_array_height, main_array_width = main_array.shape

    # The number of horizontal iterations needed for a 2d convolution with a stride of x, window size of a*b (a=width), and main array size of c*d == floor((a-c)/stride) + 1
    horizontal_iterations_needed = ((main_array_width - window_width)//stride) + 1