#importing first
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)


#Define Helper Functions
#clean_text
#removing weird symbols

#replace_urls
#text contains a lot of URLs, Replacing them with [URL] tag which we will use later

#jaccard_similarity
#It is a basic similarity measure, it just takes two bags of words and returns % of intersection in them.

def clean_text(text):
    replacements = {
        '\x89ÛÏ': '"',
        '\x89Û\x9d': '"',
        '\x89Ûª': "'",
        '\x89ÛÒ': '-',
        '\x89Û_': '',
        '\x89ÛÓ': '',
        '\x89Û¢': '',
        '\x89Ûª': '',
        '\x89Û÷': '',
        '\x89âÂ': '',

        '&gt;': '>',
        '&lt;': '<',
        '&amp;': '&',

        '\n': ' ',
    }

    for original, replacement in replacements.items():
        text = text.replace(original, replacement)

    return text


def replace_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'[URL]', text)

def jaccard_similarity(s1, s2):
    set1 = set(s1.split())
    set2 = set(s2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

#Remove Duplicates
def remove_duplicates(data):
    similarity_threshold = 0.75
    duplicates = set()

    # It will run for ca 5 minutes
    for i in range(len(data)):
        if i in duplicates:
            continue

        for j in range(i + 1, len(data)):
            if j in duplicates:
                continue
            similarity = jaccard_similarity(data.loc[i, 'text'], data.loc[j, 'text'])
            if similarity > similarity_threshold:
                duplicates.add(j)

    return data.drop(duplicates).reset_index(drop=True)

#Smaller Fixes & Feature Engineering

#fix_keywords_inplace
#It replaces encoded space %20 with a real space

#aggregate_location_inplace
#Let's lowercase our location and try to merge some more common values which are clearly the same.
#For others which appear only once per dataset, let's simply replace it with [something].
#This will allow us to significantly reduce number of classes which might be useful
#if we will try to train something like XGBoost.

#extract_url_feature_inplace
#Remember we replaced URLs with [URL]?
#Now we are creating a separate column URL and removing the useless [URL] tag.
# It can also be used in some kind of classifier.

def fix_keyword_inplace(data):
    data['keyword'] = data['keyword'].apply(
        lambda x: x.replace('%20', ' ') if pd.notna(x) and isinstance(x, str) else x)


def extract_url_feature_inplace(data):
    data['text'] = data['text'].apply(clean_text).apply(replace_urls)
    data["has_url"] = data['text'].apply(lambda text: '[URL]' in text)
    data['text'] = data['text'].apply(lambda x: x.replace('[URL]', ''))
    for i in range(10):
        data['text'] = data['text'].apply(lambda x: x.replace('  ', ' '))


def aggregate_location_inplace(data):
    mapping_dict = {
        'new york, ny': 'new york',
        'united states': 'usa',
        'nyc': 'new york',
        'london, uk': 'london',
        'london, england': 'london',
        'us': 'usa',
        'ny': 'new york',
        'earth': 'planet earth',
        'california, usa': 'california',
        'los angeles, ca': 'los angeles',
        'washington, dc': 'washington dc',
        'world': 'planet earth',
        'united kingdom': 'uk',
        'global': 'planet earth',
        'new york city': 'new york',
        'new york, usa': 'new york',
        'worldwide': 'planet earth',
        'hackney, london': 'london',
        'england': 'uk',
    }
    data['location'] = data['location'].str.lower().replace(mapping_dict)

    location_counts = data['location'].value_counts()
    singleton_values = location_counts[location_counts == 1].index.tolist()

    data['location'].replace(singleton_values, '[something]', inplace=True)




train_df = pd.read_csv("train_disaster.csv")
test_df = pd.read_csv("test_disaster.csv")

for data in [train_df, test_df]:
  extract_url_feature_inplace(data)
  fix_keyword_inplace(data)

train_df = remove_duplicates(train_df)

for data in [train_df, test_df]:
  aggregate_location_inplace(data)

print(train_df.shape)

#save the cleaned data
train_df = train_df.rename(columns={'target': 'labels'})
test_df = test_df.rename(columns={'target': 'labels'})

train_df.to_csv('train_cleaned.csv', index=False)
test_df.to_csv('test_cleaned.csv', index=False)