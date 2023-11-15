import requests
import zipfile

# Download GloVe embeddings
url = 'http://nlp.stanford.edu/data/glove.6B.zip'
file_name = 'glove.6B.zip'
extract_folder = 'glove_embeddings'

# Download GloVe embeddings zip file
response = requests.get(url)
with open(file_name, 'wb') as f:
    f.write(response.content)

# Unzip the GloVe embeddings file
with zipfile.ZipFile(file_name, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)
