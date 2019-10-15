# Data Mining Clustering Project (UPV)

## Download data set

Download [the data set from kaggle](https://www.kaggle.com/kazanova/sentiment140) and extract the file (~250MB) to the directory `data/clean`.

## Generate the pdf from markdown

Used template: https://github.com/Wandmalfarbe/pandoc-latex-template

```
pandoc -o out.pdf --from markdown --template eisvogel --toc --listings -N -V lang=en commitment.md
```