"""
Generate a word cloud from given texts (from one cluster)
"""

import glob
import os
import re
import argparse
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def plotting(word_cloud):
    """
    Method to simply plot the generated word cloud (if needed)

    :param word_cloud: generated word cloud
    :return: none
    """
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def main(plot, save, dest):
    print("### GENERATING WORD CLOUDS ###")

    for filename in glob.glob("resources/kmeans_tweets/*.txt"):

        wordcloud = WordCloud(max_words=1000, background_color='white', scale=3, relative_scaling=0.5,
                              width=500, height=400, random_state=42)

        file = open(filename, "r")
        wordcloud.generate(file.read())

        try:
            # plotting word cloud if option is chosen
            if plot is True:
                plotting(wordcloud)

            # saving word cloud to directory if option is chosen
            if save is True:
                path = "%s/wordcloud_%s.png" % (dest, re.search('\d+', os.path.basename(filename))[0])
                wordcloud.to_file(path)
                print(f"INFO: World cloud saved to {path}")
        except Exception as e:
            print(f"EXCEPTION: Error on plotting / saving word cloud ({filename}):", e)

    print("### ENDED GENERATING WORD CLOUDS ###")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run k-means clustering')
    # TODO: Set project structure / file paths
    parser.add_argument("-p", "--plot", dest="plot", default=False, type=bool,
                        help="Plot word cloud on/off")
    parser.add_argument("-s", "--save", dest="save", default=False, type=bool,
                        help="Save word cloud on/off")
    parser.add_argument("-d", "--dest", dest="dest", default=".",
                        help="Destination directory for saving word clouds")
    args = parser.parse_args()

    main(args.plot, args.save, args.dest)
