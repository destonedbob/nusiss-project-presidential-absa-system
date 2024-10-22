from preprocessor.preprocess import check_last_char_punctuation

import spacy
import pandas as pd
import numpy as np

TRUMP_ENTITY_MENTIONS = './data/entity_mentions/trump_entity_mentions.csv'
KAMALA_ENTITY_MENTIONS = './data/entity_mentions/kamala_entity_mentions.csv'

# coref_nlp = spacy.load("en_core_web_trf")
coref_nlp = spacy.load("en_coreference_web_trf")
trump_entity_list = set(pd.read_csv(TRUMP_ENTITY_MENTIONS).entity_names.values.tolist())
kamala_entity_list = set(pd.read_csv(KAMALA_ENTITY_MENTIONS).entity_names.values.tolist())

def get_coref(text):    
    result = []
    doc = coref_nlp(text)
    for cluster, values in doc.spans.items():
        collected_values = []
        for value in values:
            collected_values.append(value)
        result.append(collected_values)
    
    return result

def resolve_references(row) -> str:
    """ """
    text = row['combined_test']
    parent_comment = row['parent_comment']
    if check_last_char_punctuation(parent_comment):
        prev_comment_len = len(parent_comment) + 1
    else:
        prev_comment_len = len(parent_comment) + 2
    doc = coref_nlp(text)
    # token.idx : token.text
    token_mention_mapper = {}
    output_string = ""
    clusters = [
        val for key, val in doc.spans.items() if key.startswith("coref_cluster")
    ]

    combined_set = list(trump_entity_list.union(kamala_entity_list))

    for idx, cluster in enumerate(clusters):
        found = False
        for mention_idx, mention in enumerate(cluster):
            if any([True for val in combined_set if val.lower() in mention.text.lower()]) and found == False:
                # Handle missed
                for prev_missed_mentions in list(cluster)[:mention_idx]:
                    token_mention_mapper[prev_missed_mentions[0].idx] = mention.text + mention[0].whitespace_
                    for token in prev_missed_mentions[1:]:
                        token_mention_mapper[token.idx] = ""

                key_mention = mention
                found = True

            # Handle current
            if found:
                token_mention_mapper[mention[0].idx] = key_mention.text + mention[0].whitespace_
                for token in mention[1:]:
                    token_mention_mapper[token.idx] = ""

    # Iterate through every token in the Doc
    for token in doc:
        # Check if token exists in token_mention_mapper
        if token.idx in token_mention_mapper:
            output_string += token_mention_mapper[token.idx]
        # Else add original token text
        else:
            output_string += token.text + token.whitespace_

    try:
        return output_string.split('[NEXT_COMMENT] ')[1]
    except:
        print(f'Error in output string: {output_string}')
        return output_string
    # return output_string[prev_comment_len:]

def contains_entity_mention(text, entity_mentions):
    lowered_text = text.lower()
    for ent in entity_mentions:
        if ent.lower() in lowered_text:
            return 1
    
    return 0

def coref_resolve(df1):
    results = pd.DataFrame()
    num_na = 0
    for level in sorted(df1.level.unique()):

        temp_df = df1[df1.level == level]

        if level != 1:
            comment_dictionary = results[results.level == level - 1].set_index('comment_id')['comment_after_coref'].to_dict()
            # temp_df['parent_comment'] = temp_df.apply(lambda x: comment_dictionary[x['parent_comment_id']] if x['parent_comment'] != '[deleted]' else x['parent_comment'], axis=1)
            temp_df['parent_comment'] = np.where(temp_df['parent_comment'] != '[deleted]',  temp_df['parent_comment_id'].map(comment_dictionary), temp_df['parent_comment'])
            temp_df['parent_comment'] =  temp_df['parent_comment'].fillna('')
            
            num_na += temp_df['parent_comment'].isna().sum()

        temp_df['combined_test'] = np.where(
                                temp_df['parent_comment'].apply(check_last_char_punctuation), 
                                temp_df['parent_comment'] + ' [NEXT_COMMENT] ' + temp_df['comment'],
                                temp_df['parent_comment'] + '. [NEXT_COMMENT] ' + temp_df['comment']
                                )

        temp_df['comment_after_coref'] = temp_df.apply(resolve_references, axis=1)

        results = pd.concat([results, temp_df])

        results['contains_trump_mentions'] = results['comment'].apply(lambda text: contains_entity_mention(text, trump_entity_list)) # Shouldn't be done here, but after splitting to sentence level. will redo in later code
        results['contains_kamala_mentions'] = results['comment'].apply(lambda text: contains_entity_mention(text, kamala_entity_list)) # Shouldn't be done here, but after splitting to sentence level. will redo in later code
        results = results.drop('combined_test', axis=1)
    return results