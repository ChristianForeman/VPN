import argparse
import matplotlib.pyplot as plt

def plot_loss(filename, stride):
    print ("THe stride is ", stride)
    # Read data from the text file
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Extract data from columns
    epochs = []
    training_loss = []
    testing_loss = []

    for line in lines:
        if not line[0].isdigit():
            continue

        epoch, train_loss, test_loss = map(float, line.strip().split())
        epochs.append(epoch)
        training_loss.append(train_loss)
        testing_loss.append(test_loss)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(epochs[::stride], training_loss[::stride], label='Training Loss', marker='o')
    plt.plot(epochs[::stride], testing_loss[::stride], label='Validation Loss', marker='o')
    
    plt.title('Training and Validation Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training and testing loss vs epoch')
    parser.add_argument('-f', '--filename', type=str, help='Path to the text file containing epoch, training loss, and testing loss columns')
    parser.add_argument('-s', '--stride', default=2, type=int, help='Path to the text file containing epoch, training loss, and testing loss columns')

    args = parser.parse_args()
    plot_loss(args.filename, args.stride)