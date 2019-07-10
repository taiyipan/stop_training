from tensorflow import keras 
# define stop training callback
class StopTraining(keras.callbacks.Callback):
    def __init__(self, desired_accuracy = 0.95):
        self._desired_accuracy = desired_accuracy
    def on_epoch_end(self, epoch, logs = {}):
        if logs.get('accuracy') > self._desired_accuracy:
            print('\nReached {}% accuracy so cancelling training'.format(self._desired_accuracy * 100))
            self.model.stop_training = True
