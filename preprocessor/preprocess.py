import string
import re

from utilities.util import create_folder_if_not_exists, convert_datetime

import numpy as np
import pandas as pd
import html

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

from huggingface_hub import hf_hub_download
import fasttext
import unicodedata

lang_model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
lang_model = fasttext.load_model(lang_model_path)

def rename_df_cols(comment_df, dataset):
    if dataset == 'reddit_comments':
      remaps = {
        'Post Title': 'post_title',
        'Post ID': 'post_id',
        'Comment ID': 'comment_id',
        'Timestamp': 'comment_timestamp',
        'Votes': 'number_of_comment_votes',
        'Comment': 'comment',
        'Parent Comment ID': 'parent_comment_id'
        }
    elif dataset == 'reddit_posts':
      remaps = {
        'votes': 'number_of_post_votes',
        'created_utc': 'post_timestamp',
        'num_comments': 'post_num_comments'
        }
    elif dataset == 'youtube':
      remaps = {
        'video_title': 'post_title',
        'video_id': 'post_id',
        'comment_date': 'comment_timestamp',
        'number_of_likes': 'number_of_comment_votes',
        'Comment': 'comment',
        'video_post_date': 'post_timestamp'
        }
    else:
      raise ValueError('Invalid dataset name')

    return comment_df.rename(columns=remaps, errors='ignore')

def combine_reddit_comment_and_post_df(comment_df, post_df):
  result = comment_df.copy()
  
  result = result.merge(post_df.drop('post_title', axis=1), how='left', on='post_id')

  parent_comment_lookup = result[['comment_id', 'comment']].rename(columns={'comment': 'parent_comment'})
  result = result.merge(parent_comment_lookup, how='left', left_on='parent_comment_id', right_on='comment_id')
  result = result.drop(columns=['comment_id_y']).rename(columns={'comment_id_x': 'comment_id'})


  # # Add in parent comment
  # result = result.merge(result[['parent_comment_id', 'comment']].rename(columns={'comment':'parent_comment'}),
  #                       how='left', left_on='comment_id', right_on='parent_comment_id')\
  #                .drop('parent_comment_id_y', axis=1)\
  #                .rename(columns={'parent_comment_id_x':'parent_comment_id'})

  return result

def set_post_title_as_parent_comment_if_na(df):
  result = df.copy()
  result['parent_comment'] = result['parent_comment'].fillna(result['post_title'])
  result['parent_comment'] = result['parent_comment'].fillna(result['post_title'])
  return result

def unify_youtube_and_reddit_comments(reddit_df, youtube_df):
  reddit_df_col_to_drop = ['number_of_post_votes', 'post_body', 'post_num_comments']
  ordered_columns_for_final_df = [
        'post_id',
        'post_title',
        'post_timestamp',
        'parent_comment_id',
        'parent_comment',
        'comment_id',
        'comment',
        'comment_timestamp',
        'number_of_comment_votes',
      ]

  reddit_df_copy = reddit_df.drop(reddit_df_col_to_drop, axis=1).copy()[ordered_columns_for_final_df]
  reddit_df_copy['platform'] = 'Reddit'
  reddit_df_copy['post_timestamp'] = pd.to_datetime(reddit_df_copy['post_timestamp'], format='%d-%b-%Y', utc=True)
  reddit_df_copy['comment_timestamp'] = pd.to_datetime(reddit_df_copy['comment_timestamp'], format='%d-%b-%Y', utc=True)
  youtube_df_copy = youtube_df.copy()[ordered_columns_for_final_df]
  youtube_df_copy['platform'] = 'Youtube'
  youtube_df_copy['parent_comment'] = np.where(youtube_df_copy['parent_comment'] == 'N/A', np.nan, youtube_df_copy['parent_comment'])
  youtube_df_copy['parent_comment_id'] = np.where(youtube_df_copy['parent_comment_id'] == 'N/A', np.nan, youtube_df_copy['parent_comment_id'])
  youtube_df_copy['post_timestamp'] = pd.to_datetime(youtube_df_copy['post_timestamp'], format='ISO8601', utc=True).dt.normalize()
  youtube_df_copy['comment_timestamp'] = pd.to_datetime(youtube_df_copy['comment_timestamp'], format='ISO8601', utc=True).dt.normalize()
  return pd.concat([reddit_df_copy, youtube_df_copy])


def is_english(text):
    predictions = lang_model.predict(text.lower().replace('\n', ''))
    first_pred = predictions[0][0]
    return  first_pred == '__label__eng_Latn' 


def clean_text(text):
    # Normalize to NFC
    normalized_text = unicodedata.normalize('NFC', text)
    # Remove non-printable and formatting characters
    cleaned_text = ''.join(c for c in normalized_text if not unicodedata.category(c).startswith('Cf'))
    return cleaned_text


def check_last_char_punctuation(text):
    if not text.strip():  # In case of empty strings
        return False
    return text.strip()[-1] in string.punctuation


def replace_special_chars(text):
    replacements = {
      '“': '"', '”': '"', '‘': "'", '’': "'", 
      '–': '-', '…': '...', #'•': '-', 
      # '©': '(c)', '®': '(r)', '™': '(tm)'
    }
      
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


def assign_level(df):
    '''Assign level to the comment, comments without parent comments are level 1, their child is n+1.'''
    df['level'] = None
    
    df.loc[df['parent_comment_id'].isna(), 'level'] = 1
    # For those whose parent comment cant be found, we assume they are level 1
    df.loc[~df['parent_comment_id'].isin(df['comment_id'].unique().tolist()), 'level'] = 1

    while df['level'].isna().sum() > 0:
        current_level = df.level.max()
        current_level_list = df[df.level == current_level].comment_id.unique().tolist()
        df.loc[df.parent_comment_id.isin(current_level_list), 'level'] = current_level + 1
        current_level += 1

    return df


def preprocess_dataset(combined_df):
  # Remove unwanted NA is any
  result = combined_df[~combined_df.comment.isna()]

  # Fix formatting and encoding issues
  result['post_title'] = result['post_title'].apply(lambda x: html.unescape(x))
  result['parent_comment'] = result['parent_comment'].apply(lambda x: html.unescape(x) if not pd.isnull(x) else np.nan)
  result['comment'] = result['comment'].apply(lambda x: html.unescape(x))
  result['parent_comment'] = result['parent_comment'].fillna('')

  # Remove non english text
  result['is_english'] = result['comment'].apply(is_english)
  result = result[result['is_english']].drop(['is_english'], axis=1)
  result['comment'] = result['comment'].apply(clean_text)
  result = result.applymap(lambda x: replace_special_chars(x) if isinstance(x, str) else x)

  # Remove non human readable characters
  result = result.applymap(lambda x: re.sub(r'[\x00-\x1f\x7f-\x9f]', '', x) if isinstance(x, str) else x)
  
  # Replace URLs
  url_pattern = r'http[s]?://\S+'
  result['comment'] = result['comment'].str.replace(url_pattern, '[URL]', regex=True)

  result = assign_level(result)
  return result

def split_comments_to_sentence(df):
    rows = []

    for idx, row in df.iterrows():
        try:
          sentences = sent_tokenize(row['comment'])
        except Exception as e:
          print(row)
          print(row['comment'])
          raise e
        previous_sentence = None
        for sentence in sentences:
            rows.append({
                'comment_id': row['comment_id'],
                'post_id': row['post_id'],
                'post_title': row['post_title'],
                'post_timestamp': row['post_timestamp'],
                'parent_comment_id': row['parent_comment_id'],
                'parent_comment': row['parent_comment'],
                'comment_id': row['comment_id'],
                'comment': row['comment'],
                'comment_timestamp': row['comment_timestamp'],
                'number_of_comment_votes': row['number_of_comment_votes'],
                'sentence': sentence,
                'previous_sentence': previous_sentence
            })
            previous_sentence = sentence

    return pd.DataFrame(rows)
