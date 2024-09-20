import re
import matplotlib.pyplot as plt

# Define regex pattern for validation log entries
val_pattern = re.compile(r"""
    \[(?P<timestamp>[\d\- :]+)\]      # Timestamp
    \s+\[CRIT\]                       # Log level
    \s+\[Val\]\s+Epoch:\s+(?P<epoch>\d+),  # Epoch
    \s+c_loss:\s+(?P<c_loss>[\d\.]+), # c_loss
    \s+f_loss:\s+(?P<f_loss>[\d\.]+), # f_loss
    \s+loss:\s+(?P<loss>[\d\.]+),     # loss
    \s+PIR:\s+(?P<PIR>[\d\.]+),       # PIR
    \s+IR:\s+(?P<IR>[\d\.]+),         # IR
    \s+RRE:\s+(?P<RRE>[\d\.]+),       # RRE
    \s+RTE:\s+(?P<RTE>[\d\.]+),       # RTE
    \s+RR:\s+(?P<RR>[\d\.]+),         # RR
    \s+time:\s+[\d\.]+s/[\d\.]+s      # time
""", re.VERBOSE)

# Function to read log and extract validation entries
def extract_val_data(log_file):
    validation_data = []

    with open(log_file, 'r') as file:
        for line in file:
            match = val_pattern.search(line)
            if match:
                data = match.groupdict()
                validation_data.append({
                    'epoch': int(data['epoch']),
                    'loss': float(data['loss']),
                    'RRE': float(data['RRE']),
                    'RTE': float(data['RTE'])
                })

    return validation_data

# Graphing function
def plot_validation_data(validation_data):
    epochs = [entry['epoch'] for entry in validation_data]
    losses = [entry['loss'] for entry in validation_data]
    RREs = [entry['RRE'] for entry in validation_data]
    RTEs = [entry['RTE'] for entry in validation_data]

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(epochs, losses, marker='o', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(epochs, RREs, marker='o', color='orange', label='RRE')
    plt.xlabel('Epoch')
    plt.ylabel('RRE')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(epochs, RTEs, marker='o', color='green', label='RTE')
    plt.xlabel('Epoch')
    plt.ylabel('RTE')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('long_log.png')


log_file = '/project/bli4/maps/wacv/GeoTransformer/weights/ORFD/long_log.log'  # Replace with your log file path
validation_data = extract_val_data(log_file)
plot_validation_data(validation_data)