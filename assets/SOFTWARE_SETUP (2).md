# Minimal software setup

R is preferred because the laboratory material uses R, but R and Python are marked to the same standard.

## R

The model code needs base R, `caret` (`createDataPartition()` for the stratified
80/20 split) and `MASS` (`polr()` for ordered logit/probit). Knitting also needs
`knitr` and `rmarkdown`.

```r
install.packages(c("caret", "MASS", "knitr", "rmarkdown"))
```

## Python

Use Python 3.10 or later and install the packages in `requirements_python.txt`.

```bash
python -m pip install -r requirements_python.txt
```

`statsmodels.miscmodels.ordinal_model.OrderedModel` is used for ordered logit and ordered probit. Its explanatory matrix must not contain an ordinary intercept because the model estimates cutpoints.
`sklearn.model_selection.train_test_split()` can create the required
outcome-stratified 80/20 split.

## Rendering to HTML or PDF

HTML output does not require LaTeX. PDF output from R Markdown normally requires a LaTeX installation such as TinyTeX.

```r
install.packages("tinytex")
tinytex::install_tinytex()
```
