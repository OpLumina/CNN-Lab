import numpy as np
random_array = np.random.rand(28, 28)

def relu(x: float) -> float:
    return max(0, x)


def extract_array_from_array(x, y, width, height, array):
    extracted_array = array[y:y+height, x:x+width].copy()
    return extracted_array

"""
E.G. Extract the 3x3 matrix that is at 1,1 from A1
extract_array_from_array(1,1,3,3,A1)
A1 = 
[x,x,x,x]
[x,a,a,a]
[x,a,a,a]
[x,a,a,a]
extracted array =
[a,a,a]
[a,a,a]
[a,a,a]

Test:
random_array = np.random.rand(4, 5)
extracted_array = extract_array_from_array(1,1,3,3,random_array)
print(extracted_array)
print(random_array)


Note: array.shape returns height,width not width,height
Example:
random_array = np.random.rand(4, 5)
y,x = random_array.shape
print(x,y)
print(random_array)
"""










def feature_map(filter_array,main_array,stride,bias,total_iterations_needed="placeholder",horizontal_iterations_needed="placeholder",vertical_iterations_needed="placeholder"):
    feature_map = np.array([])

    window_height, window_width = filter_array.shape
    main_array_height, main_array_width = main_array.shape

    if horizontal_iterations_needed == "placeholder":
        horizontal_iterations_needed = ((main_array_width - window_width) // stride) + 1
    if vertical_iterations_needed == "placeholder":
        vertical_iterations_needed = ((main_array_height - window_height) // stride) + 1
    if total_iterations_needed == "placeholder":
        total_iterations_needed = horizontal_iterations_needed * vertical_iterations_needed

    for y in range(0, vertical_iterations_needed * stride, stride):
        for x in range(0, horizontal_iterations_needed * stride, stride):
            extracted_main_array = extract_array_from_array(x, y, window_width, window_height, main_array)
            feature_map_element = relu(np.einsum('ij,ij', extracted_main_array, filter_array) + bias)
            feature_map = np.append(feature_map, feature_map_element)

    feature_map = np.reshape(feature_map,(vertical_iterations_needed,horizontal_iterations_needed))
    return feature_map


"""
# test
random_array = np.random.randint(-20, 10, size=(6, 6))
random_filter_array = np.random.randint(-20, 100, size=(3, 3))
feature_map_test = feature_map(random_filter_array, random_array, 1, 0)
print("Feature Map:\n",feature_map_test)
print("Main Array:\n",random_array)
print("Filter:\n", random_filter_array)
"""



def maxpool(main_array,window_size=2,stride=2,horizontal_iterations_needed=None,vertical_iterations_needed=None,total_iterations_needed=None,):
    h, w = main_array.shape
    if horizontal_iterations_needed is None:
        horizontal_iterations_needed = (w - window_size) // stride + 1
    if vertical_iterations_needed   is None:
        vertical_iterations_needed   = (h - window_size) // stride + 1
    if total_iterations_needed is None:
        total_iterations_needed = horizontal_iterations_needed * vertical_iterations_needed

    pool_vals = []

    for y in range(vertical_iterations_needed):
        for x in range(horizontal_iterations_needed):
            x = x * stride
            y = y * stride

            window = extract_array_from_array(x, y, window_size, window_size, main_array)
            pool_vals.append(window.max())

    pool_matrix = np.array(pool_vals, dtype=main_array.dtype)
    pool_matrix = pool_matrix.reshape(vertical_iterations_needed,horizontal_iterations_needed)

    return pool_matrix
