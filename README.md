# Data Mining Clustering Project (UPV)

## Download data set

Download [the data set from kaggle](https://www.kaggle.com/kazanova/sentiment140) and extract the file (~250MB) to the directory `resources/raw`.

## Generate the pdf from markdown

Used template: https://github.com/Wandmalfarbe/pandoc-latex-template

```
pandoc -o out.pdf --from markdown --template eisvogel --toc --listings -N -V lang=en commitment.md
```

## Clustering on small dataset

```
python run_clustering.py -k 10 -m 2 --src resources/small/vecs.vec --dest resources/small/clustering --max_iter 10 --n_iter 20 --verbose True
```