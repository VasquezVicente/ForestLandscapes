import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

x_data = np.array([0,30,60,90,120,150,180,210,240,270,300,330])
y_data = np.array([99, 100, 100, 75, 50, 0, 0, 0, 50, 75, 99, 98])

# Calculate the first derivative (slope) between consecutive points
slope = np.diff(y_data) / np.diff(x_data)

sof_index = None
eof_index = None
for i in range(1, len(slope)):
    # Transition from non-negative slope to negative slope (SOF)
    if sof_index is None and slope[i-1] >= 0 and slope[i] < 0:
        sof_index = i
    # Transition from negative slope to non-negative slope (EOF)
    if sof_index is not None and slope[i-1] < 0 and slope[i] >= 0:
        eof_index = i
        break  # Once EOF is found, break the loop

# Get the corresponding x values for SOF and EOF
sof = x_data[sof_index]
eof = x_data[eof_index] if eof_index is not None else None  # EOF might not be found if the fall doesn't end

# Print the results
print(f"SOF (Start of Fall): {sof} months")
if eof is not None:
    print(f"EOF (End of Fall): {eof} months")
else:
    print("EOF (End of Fall) not detected.")

# Plot the data and highlight SOF and EOF points
plt.plot(x_data, y_data, label='Observed Data')
plt.scatter(sof, y_data[sof_index], color='red', label=f'SOF: {sof} months')
if eof is not None:
    plt.scatter(eof, y_data[eof_index], color='blue', label=f'EOF: {eof} months')
plt.xlabel('Time (months)')
plt.ylabel('Leaf Coverage')
plt.title('Leaf Coverage, SOF, and EOF Detection')
plt.legend()
plt.grid(True)
plt.show()