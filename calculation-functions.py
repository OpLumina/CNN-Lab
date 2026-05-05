import numpy as np
random_array = np.random.rand(28, 28)

def relu(number):
    number = n
    if n <= 0:
        n = 0
        return n
    else:
        return n


# Extracts an array from another array, x and y coordinate starting from top left of array to extract,
# width = width of array to extract, height = height of array to extract,
# array = array to extract from
"""
E.G. Extract the 3x3 matrix that is at 1,1 from A1
extract_array_from_array(1,1,3,3,A1)
A1 = 
[x,x,x,x]
[x,a,a,a]
[x,a,a,a]
[x,a,a,a]
extracted array:
[a,a,a]
[a,a,a]
[a,a,a]
"""

def extract_array_from_array(x, y, width, height, array):
    extracted_array = array[y:y+height, x:x+width].copy()
    return extracted_array

"""
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

def feature_map(filter_array, main_array, stride, bias, total_iterations_needed="placeholder", horizontal_iterations_needed="placeholder", vertical_iterations_needed="placeholder"):
    # Set an empty numpy array
    feature_map = np.array([])

    # Array sizes
    window_height, window_width = filter_array.shape
    main_array_height, main_array_width = main_array.shape

    # Iterations calcs if not set
    if horizontal_iterations_needed == "placeholder":
        horizontal_iterations_needed = ((main_array_width - window_width) // stride) + 1
    if vertical_iterations_needed == "placeholder":
        vertical_iterations_needed = ((main_array_height - window_height) // stride) + 1
    if total_iterations_needed == "placeholder":
        total_iterations_needed = horizontal_iterations_needed * vertical_iterations_needed

    # Setting original x and y placement for the extraction to top left
    x, y = 0, 0
    for i in range(total_iterations_needed):
        extracted_main_array = extract_array_from_array(x, y, window_width, window_height, main_array)

        # Element = Sum(extracted_main_array * filter_array) + bias
        feature_map_element = np.einsum('ij,ij', extracted_main_array, filter_array) + bias
        feature_map = np.append(feature_map, feature_map_element)

        # For each iteration move to the right
        # If moving to the right gets to the end, move down a stride and set x back to 0
        # If the array goes over the border with the stride, error and exit
        if x + stride < horizontal_iterations_needed * stride:
            x += stride
        elif x + stride > horizontal_iterations_needed * stride:
            print("Error: Invalid stride caused overflow in x values, terminating.")
            exit()
        else:
            x = 0
            if y + stride < vertical_iterations_needed * stride:
                y += stride
            elif y + stride > vertical_iterations_needed * stride:
                print("Error: Invalid stride caused overflow in y values – terminating.")
                exit()
            else:
                pass

    # Reshapes into the feature map (each iteration is a value)
    feature_map = np.reshape(feature_map, (vertical_iterations_needed, horizontal_iterations_needed))
    return feature_map

# test
random_array = np.random.rand(6, 6)
random_filter_array = np.random.rand(3, 3)
feature_map_test = feature_map(random_filter_array, random_array, 1, 0)
print("Feature Map:",feature_map_test)
print()
