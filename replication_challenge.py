!pip install rouge
!pip install sentence_transformers

import pandas as pd
import numpy as np
import math
from rouge import Rouge
from sentence_transformers import SentenceTransformer,util
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
import requests
from PIL import Image
from io import BytesIO
import json
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# load train dataset
train_df= pd.read_csv('/content/drive/MyDrive/data/train_mod.csv',sep='\t')

# load train dataset
val_df= pd.read_csv('/content/drive/MyDrive/data/val_mod.csv',sep='\t')

# load test dataset
test_df = pd.read_csv('/content/drive/MyDrive/data/test_mod.csv', sep='\t')

train_df.head(1)

val_df.head(1)

test_df.head(1)

""" Note here that the model originally used by INO is in the variable
 text_model_2, which has to do with the fact that it is never mentioned by INO,
 and we only found out later in the replication process which variant it was.
 It was easier to use this contraintuitive notation than to rewrite the
 whole code.
"""

text_model = SentenceTransformer('all-MiniLM-L6-v2')
text_model_2 = SentenceTransformer('paraphrase-MiniLM-L6-v2')
clip = SentenceTransformer('clip-ViT-B-32-multilingual-v1')
image_model = ResNet50(weights='imagenet', include_top=False)
rouge = Rouge()

"""Add text and image features to the datasets"""

# length feature
train_df['claim_length'] = train_df['claim'].str.len()
train_df['document_length'] = train_df['document'].str.len()

val_df['claim_length'] = val_df['claim'].str.len()
val_df['document_length'] = val_df['document'].str.len()

test_df['claim_length'] = test_df['claim'].str.len()
test_df['document_length'] = test_df['document'].str.len()

# ROUGE feature
def get_rouge(df):
  rouge1 = []
  for n, row in tqdm(df.iterrows(), total=len(df)):
    r = rouge.get_scores(row['document'], row['claim'])
    rouge1.append(r[0]['rouge-1']['r'])
  return rouge1

train_df['rouge'] = get_rouge(train_df)

val_df['rouge'] = get_rouge(val_df)

test_df['rouge'] = get_rouge(test_df)

# SBERT feature
def get_text_sim(df, model):
  chunk = 1024
  sim = []
  for i in tqdm(range(0, len(df), chunk), total = math.ceil(len(df)/chunk)):
    matrix = util.cos_sim(model.encode(df['claim'][i:i+chunk].to_numpy()),
                          model.encode(df['document'][i:i+chunk].to_numpy()))
    for row in [round(matrix[j, j].item(), 6) for j in range(len(matrix))]:
      sim.append(row)
  return sim

train_df['text_sim'] = get_text_sim(train_df, text_model)

train_df['text_sim_2'] = get_text_sim(train_df, text_model_2)

val_df['text_sim'] = get_text_sim(val_df, text_model)

val_df['text_sim_2'] = get_text_sim(val_df, text_model_2)

test_df['text_sim'] = get_text_sim(test_df, text_model)

test_df['text_sim_2'] = get_text_sim(test_df, text_model_2)

# CLIP Module
def get_embeddings(df):
  return np.hstack([clip.encode(df['claim']), clip.encode(df['document'])])

# X_train = get_embeddings(train_df)
X_train = pd.read_csv('/content/drive/MyDrive/data/clip_features.csv', header=None)
category = {
     'Support_Multimodal': 0,
     'Support_Text': 0,
     'Insufficient_Multimodal': 1,
     'Insufficient_Text': 1,
     'Refute': 2
 }
y = train_df['Category'].map(category)

mlp = MLPClassifier(hidden_layer_sizes=(100,), solver='adam', max_iter=20, verbose=True)
mlp.fit(X_train, y)

pred = mlp.predict(X_train)
train_df['text_clip_label'] = pred

X_val = get_embeddings(val_df)
pred = mlp.predict(X_val)
val_df['text_clip_label'] = pred

X_test = get_embeddings(test_df)
pred = mlp.predict(X_test)
test_df['text_clip_label'] = pred

# RESNET50
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

def download_images(df, path):
  for n, row in tqdm(df.iterrows(), total=len(df)):
    try:
      response = requests.get(row['claim_image'], headers=headers)
      response.raise_for_status()
      image = Image.open(BytesIO(response.content))
      image = image.convert('RGB')

      response = requests.get(row['document_image'], headers=headers)
      response.raise_for_status()
      image1 = Image.open(BytesIO(response.content))
      image1 = image1.convert('RGB')
    except:
      continue

    image1.save(path + 'document/' + str(int(n/1000)*1000) + '/document_img_' + str(n) + '.jpg')
    image.save(path + 'claim/' + str(int(n/1000)*1000) + '/claim_img_' + str(n) + '.jpg')

download_images(train_df, '/content/drive/MyDrive/data/images/train/')

download_images(val_df, '/content/drive/MyDrive/data/images/val/')

download_images(test_df, '/content/drive/MyDrive/data/images/test/')

def get_img_sim(size, path, dump_file_path):
  with open(dump_file_path, 'a') as sim:
    for n in tqdm(range(size), total=size, initial=0):
      try:
        p = path + 'claim/' + str(int(n/1000)*1000) + '/claim_img_' + str(n) + '.jpg'
        img = image.load_img(p, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        claim_pred = image_model.predict(img_array, verbose=0)

        p = path + 'document/' + str(int(n/1000)*1000) + '/document_img_' + str(n) + '.jpg'
        img = image.load_img(p, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        document_pred = image_model.predict(img_array, verbose=0)

        sim.write(str(n) + ', ' + str(cosine_similarity(np.mean(claim_pred, axis=(1,2)), np.mean(document_pred, axis=(1, 2)))[0][0]) + '\n')
        if n % 100 == 0:
          sim.flush()
      except:
        sim.write(str(n) + ', ' + str(0) + '\n')

get_img_sim(35000, '/content/drive/MyDrive/data/images/train/', '/content/drive/MyDrive/data/img_sim_train.csv')

img_sim_train_df = pd.read_csv('/content/drive/MyDrive/data/img_sim_train.csv', sep=',', header=None)
train_df['img_sim'] = img_sim_train_df[1]

get_img_sim(7500, '/content/drive/MyDrive/data/images/val/', '/content/drive/MyDrive/data/img_sim_val.csv')

img_sim_val_df = pd.read_csv('/content/drive/MyDrive/data/img_sim_val.csv', sep=',', header=None)
val_df['img_sim'] = img_sim_val_df[1]

get_img_sim(7500, '/content/drive/MyDrive/data/images/test/', '/content/drive/MyDrive/data/img_sim_test.csv')

img_sim_test_df = pd.read_csv('/content/drive/MyDrive/data/img_sim_test.csv', sep=',', header=None)
test_df['img_sim'] = img_sim_test_df[1]

"""Now that we have all the features, we save the datasets for future use and proceed with training the RandomForest classifier"""

train_df.to_csv('/content/drive/MyDrive/data/train_mod.csv',sep='\t',index=False,header=True)

val_df.to_csv('/content/drive/MyDrive/data/val_mod.csv',sep='\t',index=False,header=True)

test_df.to_csv('/content/drive/MyDrive/data/test_mod.csv',sep='\t',index=False,header=True)

"""











..."""

# filter out samples that we couldn't download images for

no_image_train = train_df[train_df['img_sim'] == 0]
merged = train_df.merge(no_image_train, how='outer', indicator=True)
X_train = merged[merged['_merge'] == 'left_only'].drop(columns='_merge')
print(X_train.shape)

no_image_val = val_df[val_df['img_sim'] == 0]
merged = val_df.merge(no_image_val, how='outer', indicator=True)
X_val = merged[merged['_merge'] == 'left_only'].drop(columns='_merge')
print(X_val.shape)

no_image_test = test_df[test_df['img_sim'] == 0]
merged = test_df.merge(no_image_test, how='outer', indicator=True)
X_test = merged[merged['_merge'] == 'left_only'].drop(columns='_merge')
print(X_test.shape)

X_train_2 = X_train[['claim_length', 'document_length', 'rouge', 'text_sim_2', 'text_clip_label', 'img_sim']]
X_val_2 = X_val[['claim_length', 'document_length', 'rouge', 'text_sim_2', 'text_clip_label', 'img_sim']]
X_test_2 = X_test[['claim_length', 'document_length', 'rouge', 'text_sim_2', 'text_clip_label', 'img_sim']]

y_train = X_train['Category']
X_train = X_train[['claim_length', 'document_length', 'rouge', 'text_sim', 'text_clip_label', 'img_sim']]
y_val = X_val['Category']
X_val = X_val[['claim_length', 'document_length', 'rouge', 'text_sim', 'text_clip_label', 'img_sim']]
y_test = X_test['Category']
X_test = X_test[['claim_length', 'document_length', 'rouge', 'text_sim', 'text_clip_label', 'img_sim']]

sc=StandardScaler()
sc.fit(X_train)
X_train = pd.DataFrame(sc.transform(X_train), columns=['claim_length', 'document_length', 'rouge', 'text_sim', 'text_clip_label', 'img_sim'])
X_val = pd.DataFrame(sc.transform(X_val), columns=['claim_length', 'document_length', 'rouge', 'text_sim', 'text_clip_label', 'img_sim'])
X_test = pd.DataFrame(sc.transform(X_test), columns=['claim_length', 'document_length', 'rouge', 'text_sim', 'text_clip_label', 'img_sim'])

sc.fit(X_train_2)
X_train_2 = pd.DataFrame(sc.transform(X_train_2), columns=['claim_length', 'document_length', 'rouge', 'text_sim_2', 'text_clip_label', 'img_sim'])
X_val_2 = pd.DataFrame(sc.transform(X_val_2), columns=['claim_length', 'document_length', 'rouge', 'text_sim_2', 'text_clip_label', 'img_sim'])
X_test_2 = pd.DataFrame(sc.transform(X_test_2), columns=['claim_length', 'document_length', 'rouge', 'text_sim_2', 'text_clip_label', 'img_sim'])

clf = RandomForestClassifier(n_estimators=500,max_depth=40, random_state=16)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print(f1_score(y_test, predictions, average='weighted'))
clf.fit(X_train_2,y_train)
predictions = clf.predict(X_test_2)
print(f1_score(y_test, predictions, average='weighted'))

"""Ablation experiments:

| Model Name           | Validation F1 score |
|----------------------|---------------------|
| Without Sentence BERT| 0.7926              |
| Without CLIP         | 0.7911              |
| Without ROUGE+length | 0.7709              |
| Without ResNet50     | 0.6007              |
| Baseline             | 0.6664              |
| Final model          | 0.8078              |
"""

# without sentence BERT
X_train_abl1 = X_train[['claim_length', 'document_length', 'rouge', 'text_clip_label', 'img_sim']].copy()
X_val_abl1 = X_val[['claim_length', 'document_length', 'rouge', 'text_clip_label', 'img_sim']].copy()

clf = RandomForestClassifier(n_estimators=500,max_depth=40, random_state=16)
clf = clf.fit(X_train_abl1,y_train)
predictions = clf.predict(X_val_abl1)
score = f1_score(y_val, predictions, average='weighted')
print(score)

# without CLIP
X_train_abl2 = X_train[['claim_length', 'document_length', 'rouge', 'text_sim', 'img_sim']]
X_val_abl2 = X_val[['claim_length', 'document_length', 'rouge', 'text_sim', 'img_sim']]

X_train_abl2_2 = X_train_2[['claim_length', 'document_length', 'rouge', 'text_sim_2', 'img_sim']]
X_val_abl2_2 = X_val_2[['claim_length', 'document_length', 'rouge', 'text_sim_2', 'img_sim']]

clf = RandomForestClassifier(n_estimators=500,max_depth=40, random_state=16)
clf.fit(X_train_abl2,y_train)
predictions = clf.predict(X_val_abl2)
score = f1_score(y_val, predictions, average='weighted')
print('all-MiniLM-L6-v2', score)

clf.fit(X_train_abl2_2,y_train)
predictions = clf.predict(X_val_abl2_2)
score = f1_score(y_val, predictions, average='weighted')
print('paraphrase-MiniLM-L6-v2', score)

# without rouge + length
X_train_abl3 = X_train[['text_sim', 'text_clip_label', 'img_sim']]
X_val_abl3 = X_val[['text_sim', 'text_clip_label', 'img_sim']]

X_train_abl3_2 = X_train_2[['text_sim_2', 'text_clip_label', 'img_sim']]
X_val_abl3_2 = X_val_2[['text_sim_2', 'text_clip_label', 'img_sim']]

clf = RandomForestClassifier(n_estimators=500,max_depth=40, random_state=16)
clf.fit(X_train_abl3,y_train)
predictions = clf.predict(X_val_abl3)
score = f1_score(y_val, predictions, average='weighted')
print('all-MiniLM-L6-v2', score)

clf.fit(X_train_abl3_2,y_train)
predictions = clf.predict(X_val_abl3_2)
score = f1_score(y_val, predictions, average='weighted')
print('paraphrase-MiniLM-L6-v2', score)

# without ResNet50
X_train_abl4 = X_train[['claim_length', 'document_length', 'rouge', 'text_sim', 'text_clip_label']]
X_val_abl4 = X_val[['claim_length', 'document_length', 'rouge', 'text_sim', 'text_clip_label']]

X_train_abl4_2 = X_train_2[['claim_length', 'document_length', 'rouge', 'text_sim_2', 'text_clip_label']]
X_val_abl4_2 = X_val_2[['claim_length', 'document_length', 'rouge', 'text_sim_2', 'text_clip_label']]

clf = RandomForestClassifier(n_estimators=500,max_depth=40, random_state=16)
clf.fit(X_train_abl4,y_train)
predictions = clf.predict(X_val_abl4)
score = f1_score(y_val, predictions, average='weighted')
print('all-MiniLM-L6-v2', score)

clf.fit(X_train_abl4_2,y_train)
predictions = clf.predict(X_val_abl4_2)
score = f1_score(y_val, predictions, average='weighted')
print('paraphrase-MiniLM-L6-v2', score)

# baseline (just SBERT + ResNet50)
X_train_abl5 = X_train[['text_sim', 'img_sim']]
X_val_abl5 = X_val[['text_sim', 'img_sim']]

X_train_abl5_2 = X_train_2[['text_sim_2', 'img_sim']]
X_val_abl5_2 = X_val_2[['text_sim_2', 'img_sim']]

clf = RandomForestClassifier(n_estimators=500,max_depth=40, random_state=16)
clf.fit(X_train_abl5,y_train)
predictions = clf.predict(X_val_abl5)
score = f1_score(y_val, predictions, average='weighted')
print('all-MiniLM-L6-v2', score)

clf.fit(X_train_abl5_2,y_train)
predictions = clf.predict(X_val_abl5_2)
score = f1_score(y_val, predictions, average='weighted')
print('paraphrase-MiniLM-L6-v2', score)