import jsonlines
import matplotlib.pyplot as plt
import numpy as np


def plot_xent_results():
    losses = [float(line.split()[-5][:-1]) for line in open('logs/train_loss.txt', 'r').readlines()]
    print(losses)
    plt.plot(losses)
    plt.xlabel('step')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.show()


def plot_section_lengths(train_data='json_data/train_section_slides.json'):
    huge_sections_count = 0
    huge_size = 10000
    lengths = []
    with jsonlines.open(train_data) as reader:
        for line in reader:
            if len(line['text']) > huge_size:
                huge_sections_count += 1
            lengths.append(len(line['text']))

    n, bins, patches = plt.hist(x=lengths, bins='auto', color='purple',
                                alpha=0.7, rwidth=0.85)
    print('# of sections with more than {} characters: {}'.format(huge_size, huge_sections_count))
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('section sizes')
    plt.ylabel('Frequency')
    plt.title('section lengths (# of characters)')
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.xlim(xmax=15000)

    plt.show()


# plot_xent_results()
plot_section_lengths()
