import en_core_web_md
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from spacy.util import minibatch, compounding
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import math
import spacy
from dont_patronize_me import DontPatronizeMe
import pandas as pd
import random
import os
from urllib import request
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
module_url = f"https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/master/semeval-2022/dont_patronize_me.py"
module_name = module_url.split('/')[-1]
print(f'Fetching {module_url}')
# with open("file_1.txt") as f1, open("file_2.txt") as f2
with request.urlopen(module_url) as f, open(module_name, 'w') as outf:
    a = f.read()
    outf.write(a.decode('utf-8'))
# from tqdm.auto import tqdm

# If you imported a different model, it would need to be added here
nlp = en_core_web_md.load()

# Initialize a dpm (Don't Patronize Me) object.
# It takes two areguments as input:
# (1) Path to the directory containing the training set files, which is the root directory of this notebook.
# (2) Path to the test set, which will be released when the evaluation phase begins. In this example,
# we use the dataset for Subtask 1, which the code will load without labels.
dpm = DontPatronizeMe('./dontpatronizeme_v1.4/', 'dontpatronizeme_pcl.tsv')
dpm.load_task1()

data = dpm.train_task1_df


def label_PLC(row):
    if row['orig_label'] in ['0', '1']:
        return 'NonPLC'
    else:
        return 'PLC'


def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    #     random.shuffle(df) #just to be sure
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.loc[perm[:train_end]]
    validate = df.loc[perm[train_end:validate_end]]
    test = df.loc[perm[validate_end:]]
    return train, validate, test


def create_categories(x):
    if x == "PLC":
        return 1
    else:
        return 0


def training(n_iter):
    # Train model
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()  # initiate a new model with random weights
    #     pretained_weights = Path('pretrained-model-vecs/model211.bin')
    #     with pretained_weights.open("rb") as file_:
    #         textcat.model.tok2vec.from_bytes(file_.read())
        print("Training the model...")

        score_f1_best = 0
        early_stop = 0

        for i in range(n_iter):
            losses = {}
            true_labels = list()  # true label
            pdt_labels = list()  # predict label

            # batch up the examples using spaCy's minibatch
            random.shuffle(train_data)  # shuffle training data every iteration
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                           losses=losses)

            with textcat.model.use_params(optimizer.averages):
                # evaluate on valid_text, valid_label
                docs = [nlp.tokenizer(text) for text in valid_text]

                for j, doc in enumerate(textcat.pipe(docs)):
                    true_series = pd.Series(valid_cats[j])
                    true_label = true_series.idxmax()  # idxmax() is the new version of argmax()
                    true_labels.append(true_label)

                    pdt_series = pd.Series(doc.cats)
                    pdt_label = pdt_series.idxmax()  # idxmax() is the new version of argmax()
                    pdt_labels.append(pdt_label)

                score_f1 = f1_score(true_labels, pdt_labels, average='macro')
                score_ac = accuracy_score(true_labels, pdt_labels)
                precision = precision_score(
                    true_labels, pdt_labels, average='macro')
                recall = recall_score(true_labels, pdt_labels, average='macro')
                if i % print_every == 0:
                    print('loss: {:.4f} f1: {:.3f} accuracy: {:.3f} P: {:.3f} R: {:.3f}'.format(
                        losses['textcat'], score_f1, score_ac, precision, recall))

                if score_f1 > score_f1_best:
                    early_stop = 0
                    score_f1_best = score_f1
    #                 with nlp.use_params(optimizer.averages):
    #                     nlp.to_disk("plc_model")
                else:
                    early_stop += 1

                if early_stop >= not_improve:
                    print('Finished training...')
                    break

                if i == n_iter:
                    print('Finished training...')


def test_f1(threshhold):
    pred = []
    for i, row in test_df.iterrows():

        if (nlp(row['text']).cats['PLC']) <= threshhold:
            pred.append('NonPLC')
        elif (nlp(row['text']).cats['PLC']) > threshhold:
            pred.append('PLC')
    true = list(test_df['PLC_label'])
    return np.round(f1_score(true, pred, average='macro'), 2)


# start here
data['PLC_label'] = data.apply(lambda row: label_PLC(row), axis=1)
df = data
train_df, valid_df = train_test_split(df, test_size=0.2)
n_iter = 5
print_every = 1
not_improve = 4

nlp = en_core_web_md.load()
textcat = nlp.create_pipe(
    "textcat",
    config={
        "exclusive_classes": True,
        "architecture": "simple_cnn",
    }
)

nlp.add_pipe(textcat, last=True)
textcat.add_label("PLC")
textcat.add_label("NonPLC")

train_df["category"] = train_df["PLC_label"].apply(create_categories)
train_texts = train_df['text']
train_cats = [{"PLC": bool(y), "NonPLC": not bool(y)}
              for y in train_df['category']]


train_data = list(zip(train_texts, [{"cats": cats}
                                    for cats in train_cats]))

valid_df["category"] = valid_df["PLC_label"].apply(create_categories)
valid_text = valid_df['text']
valid_cats = [{"PLC": bool(y), "NonPLC": not bool(y)}
              for y in valid_df['category']]

training(n_iter)
# test news
text_1 = '''
No End in Sight for California Homeless Mess
Mental illness, addiction and the release of thousands of prisoners, all undoubtedly contribute to the current California homeless situation, but the core of the problem is supply and demand'''

text_2 = '''
Meet the Kidd Who Goes Toe to Toe With Warren Buffett
Patience, concentration and courage have allowed Wilmot Kidd to rack up one of the greatest long-term track records in the history of investing. He’s a model for how to think about, and practice, intelligent investing.
'''
print("WSJ opinion piece 1 on homeless:", text_1)
print('Model Outputs:')
print(nlp(text_1).cats)
print('='*10)

print("WSJ opinion peice 2 on a ramdom guy:", text_2)
print('Model Outputs:')
print(nlp(text_1).cats)
