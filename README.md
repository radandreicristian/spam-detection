# Spam Detection

Run: 

Create conda environment and activate:

```conda install -f requirements.txt```

Create "embeddings" folder under data.

Download fastText/GloVe embeddings in that folder.

Go to src/conf/dl_conf.yaml, update the paths to embeddings if needed.

Run data generation, training and testing script (all in one for the moment):

```python src/train_dl.py```

or (for SVC/Logistic)

```python src/train_ml.py```

The results will be output to console, best model (according to validation loss) will be saved according to the modelcheckpoint callback, and tensorboards (loss/f1 plots) will be generated in the output folder.


If there is an error regarding the fastText library (with reduce_model in the stack trace), look for util.py in fastText's source code installed via conda, and comment out lines 122-123, and add the following line:

```out_reduced = ft_model.get_output_matrix()```