"""
Generate a word cloud from given texts (from one cluster)
"""


import argparse
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def plotting(word_cloud):
    """

    :param word_cloud:
    :return:
    """
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def main(plot, save):
    # main

    # TODO: iteration
    # TODO: in which form the will the tweets be delivered (Simon/Daniel) - best way to concat?!

    text = "Hello world my little darling"

    wordcloud = WordCloud(max_words=1000, background_color='white', scale=3, relative_scaling=0.5,
                   width=500, height=400, random_state=1)
    wordcloud.generate(text)

    # plotting word cloud
    if plot is True:
        plotting(wordcloud)

    # saving word cloud to directory if
    if save is True:
        wordcloud.to_file("./wordcloud.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run k-means clustering')
    parser.add_argument("-p", "--plot", default=False, type=bool,
                        help="Plot word cloud on/off")
    parser.add_argument("-s", "--save", default=False, type=bool,
                        help="Save word cloud on/off")
    args = parser.parse_args()

    main(args.plot, args.save)
