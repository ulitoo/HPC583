import struct
import numpy as np
import matplotlib.pyplot as plt

def read_float_values_and_store(filename, rows, cols):
    values = []
    with open(filename, 'rb') as file:
        while True:
            data = file.read(4)  # Assuming 'double' is 8 bytes
            if not data:
                break
            value = struct.unpack('f', data)[0]
            values.append(value)

    # Reshape the 1D list to a 2D matrix
    matrix = [values[i:i+cols] for i in range(0, len(values), cols)]
    return matrix

# Specify the filename and matrix dimensions

rows, cols = 25, 2   # Adjust these dimensions based on your data

filename = 'Results_gpu'
double_matrix_gpu = read_float_values_and_store(filename, rows, cols)


print("\nMatrix read from binary my result file:")
for row in double_matrix_gpu:
    print(row)

# vector with : [7] [Matrix size, Residual Norm,A Norm, X Norm, Machine Epsilon, Fwd Error,Elapsed Time]
# Extract the first column from each matrix for x-axis
x_values = np.array(double_matrix_gpu)[:, 0]
Time_gpu = np.array(double_matrix_gpu)[:,1]


# Create a 2x2 multi-plot
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
#################################################################################

#################################################################################

#axs[1, 0].set_xscale('log')
#axs[1, 0].set_yscale('log')

# Create a plot for the Time of each matrix
axs[1, 0].plot(x_values,Time_gpu, label='GPU/CPU Speed')


# Add labels and legend
axs[1, 0].set_title('GPU faster than CPU by x')
axs[1, 0].set_xlabel('Matrix dimension 2^n')
axs[1, 0].set_ylabel('Times faster')
axs[1, 0].legend()
axs[1, 0].grid(True)
axs[1, 0].set_xticks(x_values, [f'{int(val)}' for val in x_values])

#################################################################################

# Show the plot
# Adjust layout
plt.tight_layout()
plt.show()

