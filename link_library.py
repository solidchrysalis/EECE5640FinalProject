import tensorflow as tf
import ctypes

class CustomGD(tf.keras.optimizers.Optimizer):
    def __init__(self, type, learning_rate=0.01, **kwargs):
        super().__init__(name, **kwargs)
        self._learning_rate = learning_rate

    def choose_type():


    def build(self, var_list):
        # This method can be used to create optimizer slots like momentum, etc.
        # For plain SGD, no extra slots are needed
        pass

    def update_step(self, grad, var, learning_rate=None):
        # Custom update rule
        if learning_rate is None:
            learning_rate = self._learning_rate

        var.assign_sub(learning_rate * grad)

    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate": self._learning_rate,
        })
        return config