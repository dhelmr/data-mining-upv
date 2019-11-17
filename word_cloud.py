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


def main(src_path, plot, save, dest):
    print("### GENERATING WORD CLOUDS ###")

    for filename in glob.glob(f"{src_path}/*.txt"):

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
                path = "%swordcloud_cluster%s.png" % (dest, re.search('\d+', os.path.basename(filename))[0])
                wordcloud.to_file(path)
                print(f"INFO: World cloud saved to {path}")
        except Exception as e:
            print(f"EXCEPTION: Error on plotting / saving word cloud ({filename}):", e)

    print("### ENDED GENERATING WORD CLOUDS ###")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run k-means clustering')
    parser.add_argument("-src", dest="path", default="resources/kmeans_tweets/",
                        help="file path to txt files containing cluster content")
    parser.add_argument("-p", dest="plot", default=False, type=bool,
                        help="plot word cloud on/off")
    parser.add_argument("-s", dest="save", default=True, type=bool,
                        help="save word cloud on/off")
    parser.add_argument("-d", dest="dest", default="resources/kmeans_tweets/wordclouds/",
                        help="destination directory for saving word clouds")
    args = parser.parse_args()

    main(args.path, args.plot, args.save, args.dest)
