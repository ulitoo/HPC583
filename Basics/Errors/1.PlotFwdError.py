import struct
import numpy as np
import matplotlib.pyplot as plt

def read_double_values_and_store(filename, rows, cols):
    values = []
    with open(filename, 'rb') as file:
        while True:
            data = file.read(8)  # Assuming 'double' is 8 bytes
            if not data:
                break
            value = struct.unpack('d', data)[0]
            values.append(value)

    # Reshape the 1D list to a 2D matrix
    matrix = [values[i:i+cols] for i in range(0, len(values), cols)]
    return matrix

# Specify the filename and matrix dimensions
filename = 'Results_lapack'
rows, cols = 10, 7   # Adjust these dimensions based on your data

# Read the double values and store in a matrix
double_matrix_lapack = read_double_values_and_store(filename, rows, cols)

filename = 'Results_mine'
double_matrix_mine = read_double_values_and_store(filename, rows, cols)

# Print the read matrix
print("\nMatrix read from binary lapack file:")
for row in double_matrix_lapack:
    print(row)

print("\nMatrix read from binary my result file:")
for row in double_matrix_mine:
    print(row)

# vector with : [7] [Matrix size, Residual Norm,A Norm, X Norm, Machine Epsilon, Fwd Error,Elapsed Time]
# Extract the first column from each matrix for x-axis
x_values = np.array(double_matrix_mine)[:, 0]

# Extract the Column from each matrix
Residual_mine = np.array(double_matrix_mine)[:,1]
Residual_lapack = np.array(double_matrix_lapack)[:,1]

NormA = np.array(double_matrix_mine)[:,2]
NormX = np.array(double_matrix_mine)[:,3]

Fwd_Error_mine = np.array(double_matrix_mine)[:,5]
Fwd_Error_lapack = np.array(double_matrix_lapack)[:,5]

Time_mine = np.array(double_matrix_mine)[:,6]
Time_lapack = np.array(double_matrix_lapack)[:,6]

# Create a 2x2 multi-plot
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
#################################################################################
axs[0, 0].set_xscale('log')
axs[0, 0].set_yscale('log')

# Create a plot for the Residual row of each matrix
axs[0, 0].plot(x_values,Residual_mine, label='Residual Norm Mine')
axs[0, 0].plot(x_values,Residual_lapack, label='Residual Norm LAPACK')

# Add labels and legend
axs[0, 0].set_title('Forward Error for my implementation vs LAPACK')
axs[0, 0].set_xlabel('Matrix dimension')
axs[0, 0].set_ylabel('Residual Norm')
axs[0, 0].legend()
axs[0, 0].set_xticks(x_values, [f'{val:.2f}' for val in x_values])
#################################################################################
axs[0, 1].set_xscale('log')
axs[0, 1].set_yscale('log')

# Create a plot for the Time row of each matrix
axs[0, 1].plot(x_values,Time_mine, label='Time Mine')
axs[0, 1].plot(x_values,Time_lapack, label='Time LAPACK')

# Add labels and legend
axs[0, 1].set_title('Time for my implementation vs LAPACK')
axs[0, 1].set_xlabel('Matrix dimension')
axs[0, 1].set_ylabel('Time in Secs')
axs[0, 1].legend()
axs[0, 1].set_xticks(x_values, [f'{val:.2f}' for val in x_values])
#################################################################################
axs[1, 0].set_xscale('log')
axs[1, 0].set_yscale('log')

# Create a plot for the ANorm row of each matrix
axs[1, 0].plot(x_values,NormA, label='Norm A')

# Add labels and legend
axs[1, 0].set_title('Norm of Matrix A')
axs[1, 0].set_xlabel('Matrix dimension')
axs[1, 0].set_ylabel('Norm')
axs[1, 0].legend()
axs[1, 0].set_xticks(x_values, [f'{val:.2f}' for val in x_values])
#################################################################################
axs[1, 1].set_xscale('log')
axs[1, 1].set_yscale('log')

# Create a plot for the Time row of each matrix
axs[1, 1].plot(x_values,NormX, label='Norm X')

# Add labels and legend
axs[1, 1].set_title('Norm of Matrix X')
axs[1, 1].set_xlabel('Matrix dimension')
axs[1, 1].set_ylabel('Norm')
axs[1, 1].legend()
axs[1, 1].set_xticks(x_values, [f'{val:.2f}' for val in x_values])

#################################################################################

# Show the plot
# Adjust layout
plt.tight_layout()
plt.show()

