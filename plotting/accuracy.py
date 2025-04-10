import matplotlib.pyplot as plt

# Data for plotting
epochs = list(range(1, 51))
loss = [
    0.543, 0.435, 0.377, 0.340, 0.317, 0.299, 0.284, 0.271, 0.255, 0.243,
    0.233, 0.221, 0.211, 0.200, 0.190, 0.181, 0.170, 0.161, 0.152, 0.141,
    0.132, 0.124, 0.113, 0.104, 0.093, 0.085, 0.076, 0.067, 0.058, 0.050,
    0.042, 0.036, 0.029, 0.025, 0.020, 0.016, 0.012, 0.009, 0.007, 0.006,
    0.005, 0.004, 0.003, 0.003, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001
]

# Create the plot
plt.plot(epochs, loss, marker='o', linestyle='-', color='b')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
