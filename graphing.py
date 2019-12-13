import sys
import matplotlib.pyplot as plt

def graph(history, to_file=None):
    if not to_file:
        sys.exit('Specify output location: to_file')
    plt.plot(history.history['crps'])
    plt.plot(history.history['val_crps'])
    plt.title('Model CRPS')
    plt.ylabel('CRPS')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(to_file)