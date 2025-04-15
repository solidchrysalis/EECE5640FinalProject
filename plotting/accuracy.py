import matplotlib.pyplot as plt

# Data for plotting
epochs = list(range(1, 51))
adam_loss_cifar_p100 = [
    0.383, 0.281, 0.240, 0.213, 0.195, 0.177, 0.163, 0.151, 0.138, 0.126,
    0.119, 0.106, 0.095, 0.088, 0.078, 0.073, 0.066, 0.064, 0.064, 0.060,
    0.054, 0.055, 0.052, 0.052, 0.056, 0.052, 0.052, 0.050, 0.054, 0.052,
    0.052, 0.051, 0.052, 0.049, 0.053, 0.055, 0.054, 0.056, 0.053, 0.051,
    0.052, 0.056, 0.046, 0.053, 0.053, 0.059, 0.054, 0.059, 0.054, 0.058
]

sgd_loss_cifar_p100 = [
    0.558, 0.475, 0.429, 0.398, 0.368, 0.350, 0.337, 0.323, 0.312, 0.305,
    0.295, 0.286, 0.279, 0.271, 0.263, 0.256, 0.249, 0.240, 0.235, 0.227,
    0.222, 0.213, 0.208, 0.202, 0.196, 0.190, 0.183, 0.179, 0.171, 0.165,
    0.159, 0.153, 0.149, 0.142, 0.137, 0.130, 0.123, 0.117, 0.113, 0.107,
    0.102, 0.096, 0.089, 0.082, 0.078, 0.072, 0.066, 0.061, 0.056, 0.052
]

adagrad_loss_cifar_p100 = [
    0.572, 0.559, 0.553, 0.550, 0.547, 0.545, 0.542, 0.541, 0.539, 0.537,
    0.536, 0.535, 0.534, 0.532, 0.533, 0.531, 0.529, 0.528, 0.527, 0.527,
    0.526, 0.526, 0.524, 0.524, 0.523, 0.523, 0.522, 0.521, 0.521, 0.520,
    0.520, 0.519, 0.519, 0.518, 0.518, 0.517, 0.517, 0.516, 0.517, 0.515,
    0.515, 0.515, 0.514, 0.515, 0.514, 0.514, 0.514, 0.513, 0.514, 0.514
]

adam_loss_cifar_v100 = [
    0.396, 0.291, 0.256, 0.227, 0.209, 0.196, 0.182, 0.166, 0.154, 0.142,
    0.130, 0.119, 0.112, 0.104, 0.099, 0.090, 0.088, 0.086, 0.083, 0.078,
    0.078, 0.073, 0.074, 0.069, 0.069, 0.069, 0.069, 0.067, 0.070, 0.068,
    0.064, 0.067, 0.065, 0.065, 0.066, 0.067, 0.070, 0.067, 0.067, 0.067,
    0.066, 0.069, 0.069, 0.067, 0.074, 0.067, 0.071, 0.074, 0.072, 0.075
]

sgd_loss_cifar_v100 = [
    0.566, 0.492, 0.438, 0.407, 0.380, 0.358, 0.342, 0.329, 0.317, 0.307,
    0.298, 0.288, 0.279, 0.270, 0.262, 0.257, 0.249, 0.242, 0.236, 0.228,
    0.222, 0.218, 0.211, 0.205, 0.198, 0.191, 0.186, 0.180, 0.173, 0.166,
    0.160, 0.156, 0.149, 0.144, 0.137, 0.132, 0.125, 0.119, 0.115, 0.107,
    0.100, 0.096, 0.090, 0.083, 0.078, 0.073, 0.068, 0.063, 0.057, 0.052
]

adagrad_loss_cifar_v100 = [
    0.378, 0.283, 0.251, 0.234, 0.220, 0.211, 0.202, 0.193, 0.186, 0.180,
    0.174, 0.168, 0.163, 0.158, 0.151, 0.148, 0.143, 0.140, 0.135, 0.131,
    0.128, 0.124, 0.120, 0.116, 0.114, 0.111, 0.108, 0.105, 0.103, 0.098,
    0.095, 0.092, 0.089, 0.087, 0.084, 0.081, 0.080, 0.076, 0.074, 0.072,
    0.070, 0.067, 0.065, 0.063, 0.061, 0.059, 0.057, 0.055, 0.054, 0.053
]

sgd_loss_fashion_p100 = [
    0.562, 0.480, 0.429, 0.397, 0.372, 0.354, 0.337, 0.326, 0.312, 0.303,
    0.295, 0.286, 0.279, 0.271, 0.265, 0.257, 0.251, 0.244, 0.238, 0.231,
    0.225, 0.219, 0.213, 0.207, 0.201, 0.196, 0.187, 0.180, 0.175, 0.170,
    0.166, 0.159, 0.154, 0.146, 0.142, 0.137, 0.130, 0.125, 0.120, 0.113,
    0.107, 0.100, 0.095, 0.090, 0.084, 0.080, 0.073, 0.069, 0.063, 0.059
]

adagrad_loss_fashion_p100 = [
    0.572, 0.559, 0.553, 0.550, 0.547, 0.545, 0.542, 0.541, 0.539, 0.537,
    0.536, 0.535, 0.534, 0.532, 0.533, 0.531, 0.529, 0.528, 0.527, 0.527,
    0.526, 0.526, 0.524, 0.524, 0.523, 0.523, 0.522, 0.521, 0.521, 0.520,
    0.520, 0.519, 0.519, 0.518, 0.518, 0.517, 0.517, 0.516, 0.517, 0.515,
    0.515, 0.515, 0.514, 0.515, 0.514, 0.514, 0.514, 0.513, 0.514, 0.514
]

adam_loss_fashion_p100 = [
    0.371, 0.280, 0.243, 0.215, 0.192, 0.170, 0.152, 0.136, 0.121, 0.112,
    0.100, 0.095, 0.085, 0.083, 0.074, 0.072, 0.070, 0.069, 0.064, 0.064,
    0.063, 0.068, 0.068, 0.065, 0.061, 0.059, 0.063, 0.065, 0.068, 0.065,
    0.069, 0.067, 0.069, 0.067, 0.067, 0.067, 0.067, 0.071, 0.065, 0.071,
    0.070, 0.071, 0.071, 0.077, 0.073, 0.071, 0.070, 0.070, 0.079, 0.080
]

sgd_loss_fashion_v100 = [
    0.565, 0.484, 0.436, 0.403, 0.374,
    0.352, 0.337, 0.324, 0.311, 0.302,
    0.293, 0.284, 0.277, 0.268, 0.261,
    0.253, 0.246, 0.239, 0.232, 0.228,
    0.221, 0.214, 0.208, 0.203, 0.194,
    0.190, 0.185, 0.177, 0.173, 0.167,
    0.162, 0.155, 0.149, 0.144, 0.137,
    0.130, 0.126, 0.120, 0.115, 0.109,
    0.104, 0.097, 0.092, 0.087, 0.080,
    0.076, 0.071, 0.066, 0.061, 0.057
]

adagrad_loss_fashion_v100 = [
    0.574, 0.570, 0.563, 0.557, 0.552,
    0.548, 0.546, 0.543, 0.542, 0.540,
    0.539, 0.538, 0.537, 0.537, 0.536,
    0.535, 0.534, 0.533, 0.533, 0.532,
    0.532, 0.531, 0.530, 0.530, 0.530,
    0.529, 0.529, 0.528, 0.528, 0.527,
    0.527, 0.527, 0.526, 0.526, 0.526,
    0.525, 0.525, 0.525, 0.524, 0.524,
    0.523, 0.524, 0.523, 0.522, 0.523,
    0.522, 0.521, 0.522, 0.522, 0.522
]

adam_loss_fashion_v100 = [
    0.389, 0.291, 0.252, 0.227, 0.210,
    0.187, 0.168, 0.154, 0.138, 0.126,
    0.111, 0.101, 0.091, 0.084, 0.081,
    0.076, 0.073, 0.068, 0.068, 0.062,
    0.062, 0.060, 0.061, 0.061, 0.056,
    0.057, 0.056, 0.062, 0.061, 0.057,
    0.056, 0.058, 0.063, 0.058, 0.063,
    0.065, 0.065, 0.064, 0.065, 0.062,
    0.059, 0.066, 0.066, 0.065, 0.065,
    0.065, 0.064, 0.065, 0.070, 0.065
]



# Set up subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Loss vs Epochs for Different Optimizers and GPUs', fontsize=16)

# Titles for the plots
titles_cifar = [
    'SGD - P100 - Cifar', 'Adam - P100 - Cifar', 'AdaGrad - P100 - Cifar',
    'SGD - V100 - Cifar', 'Adam - V100 - Cifar', 'AdaGrad - V100 - Cifar',
]

titles_fashion = [
    'SGD - P100 - FashionMNIST', 'Adam - P100 - FashionMNIST', 'AdaGrad - P100 - FashionMNIST',
    'SGD - V100 - FashionMNIST', 'Adam - V100 - FashionMNIST', 'AdaGrad - V100 - FashionMNIST',
]

# Data for each subplot
loss_lists_cifar = [
    sgd_loss_cifar_p100, adam_loss_cifar_p100, adagrad_loss_cifar_p100,
    sgd_loss_cifar_v100, adam_loss_cifar_v100, adagrad_loss_cifar_v100
]

loss_lists_fashion = [
    sgd_loss_fashion_p100, adam_loss_fashion_p100, adagrad_loss_fashion_p100,
    sgd_loss_fashion_v100, adam_loss_fashion_v100, adagrad_loss_fashion_v100
]

# Plotting
for i, ax in enumerate(axs.flat):
    ax.plot(epochs, loss_lists_cifar[i], marker='o', linestyle='-', color='b')
    ax.set_title(titles_cifar[i])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

for i, ax in enumerate(axs.flat):
    ax.plot(epochs, loss_lists_fashion[i], marker='o', linestyle='-', color='b')
    ax.set_title(titles_fashion[i])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Dummy data (replace with your actual final accuracies and kernel times)
methods = ['SGD', 'Adam', 'Adagrad']
final_accuracies_p100 = [0.70, 0.65, 0.25]  # Replace with real values
average_times_p100 = [1.2256545818414322, 1.45086010, 1.1757206662404092]     # In milliseconds, replace with real values
error_vals_p100 = [.005, .005, .005]

# Plotting final accuracies
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.bar(methods, final_accuracies_p100, color=['blue', 'green', 'orange'])
plt.ylim(0, 1)
plt.title('Final Accuracies for P100, CIFAR')
plt.ylabel('Accuracy')

# Plotting average kernel times
plt.subplot(1, 2, 2)
plt.bar(methods, average_times_p100, color=['blue', 'green', 'orange'])
plt.title('Average Kernel Execution Times, Total - P100, CIFAR')
plt.ylabel('Time (ms)')

plt.tight_layout()
plt.show()

final_accuracies_v100 = [0.67, 0.63, 0.64]  # Replace with real values
average_times_v100 = [0.520959378516624, 0.6154786304347826, 0.49646024552429663]     # Placeholder values

# Plotting final accuracies
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.bar(methods, final_accuracies_v100, color=['blue', 'green', 'orange'])
plt.ylim(0, 1)
plt.title('Final Accuracies for V100, CIFAR')
plt.ylabel('Accuracy')

# Plotting average kernel times
plt.subplot(1, 2, 2)
plt.bar(methods, average_times_v100, color=['blue', 'green', 'orange'])
plt.title('Average Kernel Execution Times - V100, CIFAR')
plt.ylabel('Time (ms)')

plt.tight_layout()
plt.show()

final_accuracies_p100 = [0.70, 0.65, 0.27]  # Replace with real values
average_times_p100 = [1.1784211, 1.4450469143222509, 1.295852812020]     # In milliseconds, replace with real values

# Plotting final accuracies
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.bar(methods, final_accuracies_p100, color=['blue', 'green', 'orange'])
plt.ylim(0, 1)
plt.title('Final Accuracies for P100, FashionMNIST')
plt.ylabel('Accuracy')

# Plotting average kernel times
plt.subplot(1, 2, 2)
plt.bar(methods, average_times_p100, color=['blue', 'green', 'orange'])
plt.title('Average Kernel Execution Times, Total - P100, FashionMNIST')
plt.ylabel('Time (ms)')

plt.tight_layout()
plt.show()

final_accuracies_v100 = [0.70, 0.69, 0.24]  # Replace with real values
average_times_v100 = [0.46778799488491046, 0.5673245319693094, 0.451896432225064]     # Placeholder values

# Plotting final accuracies
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.bar(methods, final_accuracies_v100, color=['blue', 'green', 'orange'])
plt.ylim(0, 1)
plt.title('Final Accuracies for V100, FashionMNIST')
plt.ylabel('Accuracy')

# Plotting average kernel times
plt.subplot(1, 2, 2)
plt.bar(methods, average_times_v100, color=['blue', 'green', 'orange'])
plt.title('Average Kernel Execution Times - V100, FashionMNIST')
plt.ylabel('Time (ms)')

plt.tight_layout()
plt.show()