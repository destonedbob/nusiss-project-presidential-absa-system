from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, \
                        AutoModelForSeq2SeqLM, BartForConditionalGeneration, AutoConfig
import torch
from torch import nn

import pandas as pd
import numpy as np

ENTITY_MODEL = 'destonedbob/nusiss-election-project-entity-model-distilbert-base-cased'
ASPECT_MODEL_DISTIL = './model/multilabel_aspect_distil_4epochs_lr3e-5_without_test_set_split_keep_same_sent_together.pth'
ASPECT_MODEL_SEQ2SEQ = 'destonedbob/nusiss-election-project-aspect-seq2seq-model-facebook-bart-large'
SENTIMENT_MODEL_DISTIL = './model/sentiment_model_val_acc_6162_lr4.5e-5_wtdecay_1e-4_epochs4_256_256_256_256_smoothed_weight_warmup_and_reducelr_freeze4layers.pth'
SENTIMENT_MODEL_SEQ2SEQ = 'destonedbob/nusiss-election-project-sentiment-seq2seq-model-facebook-bart-large'
DISTILBERT_BASE_CASED = 'distilbert-base-cased'


entity_idx_map = {k:v for v, k in enumerate(['kamala', 'trump', 'others'])}
idx_entity_map = {v:k for k, v in entity_idx_map.items()}
aspect_idx_map = {k:v for v, k in enumerate(['campaign', 'communication', 'competence', 'controversies',
       'ethics and integrity', 'leadership', 
       'personality trait', 'policies', 'political ideology',
       'public image', 'public service record',
       'relationships and alliances', 'voter sentiment', 'others'])}
idx_aspect_map = {v:k for k, v in aspect_idx_map.items()}
sentiment_idx_map = {k:v for v, k in enumerate(['negative', 'neutral', 'positive'])}
idx_sentiment_map = {v:k for k, v in sentiment_idx_map.items()}
idx_sentiment_map2 = {v-1:k for k, v in sentiment_idx_map.items()}

class AspectBasedSentimentModel(nn.Module):
    def __init__(self, pretrained_model_name):
        super(AspectBasedSentimentModel, self).__init__()

        num_extra_dims = 2
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.bert = AutoModel.from_pretrained(pretrained_model_name, config=self.config)

        for param in self.bert.transformer.layer[:4].parameters():  # Freeze first 4 layers
            param.requires_grad = False

        num_hidden_size = self.bert.config.hidden_size 
        self.entity_embedding = nn.Embedding(num_embeddings=3, embedding_dim=256)
        self.aspect_embedding = nn.Embedding(num_embeddings=14, embedding_dim=256)
        self.dropout = nn.Dropout(0.2)
        # self.classifier = torch.nn.Linear(num_hidden_size+num_extra_dims, 3)
        # self.classifier = torch.nn.Linear(num_hidden_size + 1028 + 1028, 3)
        self.entity_labels_embedding = nn.Embedding(num_embeddings=2, embedding_dim=256)  # Embedding for binary 0/1 values
        self.aspect_labels_embedding = nn.Embedding(num_embeddings=2, embedding_dim=256)  # Embedding for binary 0/1 values

        self.classifier = nn.Linear(num_hidden_size + 256 + 256 + (256 * 3) + (256 * 14), 514)  # First hidden layer
        self.relu = nn.ReLU()  # Activation function
        # Second linear layer (output layer)
        self.output_layer = nn.Linear(514, 3)  # Final layer to output class probabilities

    def forward(self, input_ids, attention_mask, entity_cat, aspect_cat, entity_labels, aspect_labels):
        
        hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask)  # [batch size, sequence length, hidden size]
        cls_embeddings = hidden_states.last_hidden_state[:, 0, :] # [batch size, hidden size]
        # concat = torch.cat((cls_embeddings, entity_cat.unsqueeze(1), aspect_cat.unsqueeze(1)), axis=1) # [batch size, hidden size+num extra dims]
        
        entity_embed = self.entity_embedding(entity_cat.type(torch.IntTensor).to('cuda'))
        aspect_embed = self.aspect_embedding(aspect_cat.type(torch.IntTensor).to('cuda'))
        # print((cls_embeddings.shape, entity_embed.shape, aspect_embed.shape))
        entity_labels_embed = self.entity_labels_embedding(entity_labels.type(torch.LongTensor).to('cuda')).view(entity_labels.shape[0], -1)  # Flatten [batch_size, 3, 50] to [batch_size, 150]
        aspect_labels_embed = self.aspect_labels_embedding(aspect_labels.type(torch.LongTensor).to('cuda')).view(aspect_labels.shape[0], -1)  # Flatten [batch_size, 14, 50] to [batch_size, 700]
        
        # Concatenate embeddings with CLS token output
        concat = torch.cat((cls_embeddings, entity_embed, aspect_embed, entity_labels_embed, aspect_labels_embed), axis=1)
        hidden_output = self.relu(self.classifier(self.dropout(concat)))  # [batch size, 128]

        # logits = self.output_layer(self.dropout(hidden_output))  # [batch size, num labels]
        logits = self.output_layer(hidden_output)  # [batch size, num labels]


        # logits = self.classifier(self.dropout(concat)) # [batch size, num labels]

        return logits
    

def get_entity_probabilities(texts, model, tokenizer, score=False):
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs.to('cuda')
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits)
        if not score:
            return np.array(list(map(lambda x: 1 if x > 0.65 else 0, probabilities.cpu().detach().numpy()[0].tolist())))
        else:
            return probabilities.cpu().detach().numpy()[0]
        
class MultiLabelClassifier(nn.Module):
    def __init__(self, num_labels):
        super(MultiLabelClassifier, self).__init__()
        # self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.distilbert = AutoModel.from_pretrained('distilbert-base-cased')
        self.fc1 = nn.Linear(self.distilbert.config.hidden_size + 1, 256)  # +1 for entity_ids
        self.fc2 = nn.Linear(256, num_labels)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, input_ids, attention_mask, entity_ids):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0]  # Use the [CLS] token representation
        
        entity_ids_expanded = entity_ids.view(entity_ids.size(0), -1)  # Added

        # Concatenate pooled_output with entity_ids
        # combined_output = torch.cat((pooled_output, entity_ids.unsqueeze(1)), dim=1)
        combined_output = torch.cat((pooled_output, entity_ids_expanded), dim=1)
        
        x = self.fc1(combined_output)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return torch.sigmoid(x)  # Use sigmoid for multi-label classification
    
    
def predict_distill_aspect_scores(model, tokenizer, dataframe, max_length=512):
    model.eval()  # Set the model to evaluation mode
    
    scores = []  # List to hold the scores for each row
    
    for _, row in dataframe.iterrows():
        # Tokenize the text and prepare inputs
        tokenized = tokenizer(
            row['sentence'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        # Extract the entity_id and prepare it as a LongTensor
        entity_id_tensor = torch.tensor([row['entity_id']], dtype=torch.long).to('cuda')  # Shape: (1, 1)

        # Move tokenized inputs to the device
        input_ids = tokenized['input_ids'].to('cuda')
        attention_mask = tokenized['attention_mask'].to('cuda')

        # Perform inference
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask, entity_ids=entity_id_tensor)
            scores.append(output.cpu().numpy().flatten().tolist())  # Flatten to a list of length 14

    # Convert scores to a list of lists and add to DataFrame
    dataframe['distil_aspect_scores'] = scores
    
    return dataframe


def get_distill_aspect_category(x, map):
        if sum(x) == 0:
            return ['others']
        else:
            result = []
            for idx, value in enumerate(x):
                if value == 1:
                    result.append(map[idx])
                
            return result
        

def predict_bart_aspect_model(texts, model, tokenizer, batch_size=32, return_conf=False, return_aspect_list=False):
        predictions = []

        # Tokenize texts in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize the batch
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs.to('cuda')  # Move to GPU

            if return_conf:
                with torch.no_grad():  # Disable gradient calculation
                    output_ids = model.generate(**inputs, output_scores=True, return_dict_in_generate=True)
                    predictions.append(output_ids.sequences_scores.cpu().numpy())  # Collect scores
            
            else:
                with torch.no_grad():  # Disable gradient calculation
                    output_ids = model.generate(**inputs)

                # Decode the batch of generated sequences
                output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                if return_aspect_list:
                    for output_text in output_texts:
                        aspect_lst = output_text.split(';')
                        aspect_lst_result = [0] * len(aspect_idx_map)
                        for aspect in aspect_lst:
                            processed_aspect = aspect.lower().strip()
                            aspect_id = aspect_idx_map.get(processed_aspect, 9999)
                            if aspect_id == 9999:
                                print(processed_aspect)
                                continue
                            aspect_lst_result[aspect_id] = 1

                        predictions.append(aspect_lst_result)
                else:
                    predictions.extend(output_texts)  # Append the decoded texts

        return predictions


def get_aspect_category_bart(x, map):
        if type(x) != list:
            return np.nan
        
        if sum(x) == 0:
            return ['others']
        else:
            result = []
            for idx, value in enumerate(x):
                if value == 1:
                    result.append(map[idx])
                
            return result



def create_sentence_for_sentiment_seq2seq(row):
        return f"entity of interest: {row['entity_category'].replace('others', 'neither trump nor kamala')} [SEP] aspect of interest: {row['final_aspect_categories']} [SEP] {row['sentence']}"


def predict_sentiment_bart_model(texts, model, tokenizer, batch_size=32, return_conf=False, return_both=False):
    predictions = []
    confidences = []

    # Tokenize texts in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize the batch
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = inputs.to('cuda')  # Move to GPU

        if return_both:
            with torch.no_grad():  # Disable gradient calculation
                output_ids = model.generate(**inputs, output_scores=True, return_dict_in_generate=True)
                confidences.extend(output_ids.sequences_scores.cpu().tolist())

            output_texts = tokenizer.batch_decode(output_ids.sequences, skip_special_tokens=True)
            for output_text in output_texts:
                try:
                    # Extract label and map it to sentiment index
                    final_label = sentiment_idx_map[output_text.split(': ')[1]] - 1
                except:
                    print(output_text)
                    final_label = 0
                predictions.append(final_label)
            
        elif return_conf:
            with torch.no_grad():  # Disable gradient calculation
                output_ids = model.generate(**inputs, output_scores=True, return_dict_in_generate=True)
                # Collect sequence scores for confidence
                predictions.extend(output_ids.sequences_scores.cpu().tolist())
        else:
            with torch.no_grad():  # Disable gradient calculation
                output_ids = model.generate(**inputs)

            # Decode the batch of generated sequences
            output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            for output_text in output_texts:
                try:
                    # Extract label and map it to sentiment index
                    final_label = sentiment_idx_map[output_text.split(': ')[1]] - 1
                except:
                    print(output_text)
                    final_label = 0
                predictions.append(final_label)

    if return_both:
        return predictions, confidences
    else:
        return predictions
        


def predict_sentiment_distil(row, tokenizer, model, return_conf=False, return_both=False):
        model.eval()
        inputs = tokenizer(row['sentence'], return_tensors='pt', truncation=True, padding=True, max_length=512)
        input_ids = inputs['input_ids'].to('cuda')
        attention_mask = inputs['attention_mask'].to('cuda')
        entity_cat = torch.tensor([row['entity_id']]).to('cuda')
        aspect_cat = torch.tensor([row['final_aspect_ids']]).to('cuda')
        entity_labels = torch.tensor([row['entity_ids']]).to('cuda')
        aspect_labels = torch.tensor([row['final_aspect_labels']]).to('cuda')
        
        with torch.no_grad():
            logits = model(input_ids, attention_mask, entity_cat, aspect_cat, entity_labels, aspect_labels)
        
        if return_both:
            predicted_label = torch.argmax(logits, dim=-1).item()
            return (predicted_label - 1, torch.max(logits).item())
        
        if return_conf:
            return torch.max(logits).item()
        else:
            predicted_label = torch.argmax(logits, dim=-1).item()
            return predicted_label - 1
        

def get_final_sentiment_pred(row):
        seq2seq_conf = row['seq2seq_sentiment_confidence_score']
        distil_conf = row['distil_sentiment_confidence_score']

        if distil_conf >= 0.85: # 0.6
            return row['distil_sentiment_prediction']
        elif seq2seq_conf >= -0.36: # -0.32499999999999996
            return row['seq2seq_sentiment_prediction']
        else:
            return row['distil_sentiment_prediction']
        

def predict_with_models(df):
    
    result = df.copy()
    original_columns = result.columns.tolist()

    # Entity Extraction
    model = AutoModelForSequenceClassification.from_pretrained(ENTITY_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(ENTITY_MODEL)
    model.to('cuda')
        
    result['entity_ids'] = result.sentence.apply(lambda x: get_entity_probabilities(x, model, tokenizer))
    df_columns = result.columns.tolist()
    # Expand the DataFrame for each entity
    expanded_rows = []
    for index, row in result.iterrows():
        entity_labels = row['entity_ids']
        if entity_labels[0] == 1:
            dict_row = dict()
            for col in result.columns:
                dict_row[col] = row[col]
                dict_row['entity_category'] = 'kamala'
                dict_row['entity_id'] = entity_idx_map['kamala']

            expanded_rows.append(dict_row)

        if entity_labels[1] == 1:
            dict_row = dict()
            for col in result.columns:
                dict_row[col] = row[col]
                dict_row['entity_category'] = 'trump'
                dict_row['entity_id'] = entity_idx_map['trump']
            expanded_rows.append(dict_row)

        if entity_labels[2] == 1 or sum(entity_labels) == 0:
            dict_row = dict()
            for col in result.columns:
                dict_row[col] = row[col]
                dict_row['entity_category'] = 'others'
                dict_row['entity_id'] = entity_idx_map['others']
            expanded_rows.append(dict_row)

    # Create a new DataFrame from expanded rows
    result = pd.DataFrame(expanded_rows)
    result = result[df_columns + ['entity_category', 'entity_id']]

    # Aspect Extraction (Model 1)
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_BASE_CASED)
    num_aspects = 13


    model = MultiLabelClassifier(num_labels=num_aspects).to('cuda')
    model.load_state_dict(torch.load(ASPECT_MODEL_DISTIL))

    result = predict_distill_aspect_scores(model, tokenizer, result)
    result['distil_aspect_labels'] = result.distil_aspect_scores.apply(lambda x: np.where(np.array(x) >= 0.35, 1.0, 0.0)) # Returns list of length 13, if all 0 then its others.
    result['distil_aspect_categories'] = result['distil_aspect_labels'].apply(lambda x: get_distill_aspect_category(x, idx_aspect_map))
    result['sentence2'] = result.apply(lambda row: 'entity of interest: ' + row['entity_category'].replace('others', 'neither trump nor kamala') + ' [SEP] ' + row['sentence'], axis=1)

    # Aspect Extraction (Model 2)
    result['sentence2'] = result.apply(lambda row: 'entity of interest: ' + row['entity_category'].replace('others', 'neither trump nor kamala') + ' [SEP] ' + row['sentence'], axis=1)
    tokenizer = AutoTokenizer.from_pretrained(ASPECT_MODEL_SEQ2SEQ)
    model = AutoModelForSeq2SeqLM.from_pretrained(ASPECT_MODEL_SEQ2SEQ)
    model.to('cuda')
    
    mask = result['distil_aspect_categories'].apply(lambda x: x == ['others'])

    # Get the sentences where the condition is met
    texts_to_predict = result.loc[mask, 'sentence2'].tolist()

    # Call the combined predict function
    if texts_to_predict:
        predicted_aspect_lists = predict_bart_aspect_model(texts_to_predict, model, tokenizer, batch_size=32, return_aspect_list=True)
        # Assign the predictions back to the relevant rows using .loc
        result.loc[mask, 'bart_aspect_labels'] = pd.Series(predicted_aspect_lists, index=result[mask].index)
        
    result['bart_aspect_categories'] = result['bart_aspect_labels'].apply(lambda x: get_aspect_category_bart(x, idx_aspect_map))
    result['final_aspect_categories'] = result['bart_aspect_categories'].fillna(result['distil_aspect_categories'])
    result['final_aspect_labels'] = result['bart_aspect_labels'].fillna(result['distil_aspect_labels'].apply(lambda x: [int(each) for each in x] + [1] if sum(x) == 0 else [int(each) for each in x] + [0]))
    result = result.explode('final_aspect_categories').reset_index(drop=True)



    # Sentiment Prediction (Model 1)
    result['final_aspect_ids'] = result['final_aspect_categories'].apply(lambda x: aspect_idx_map[x])

    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_BASE_CASED)
    # torch.save(model, 'sentiment_model_val_acc_6162_lr4.5e-5_wtdecay_1e-4_epochs4_256_256_256_256_warmup_and_reducelr.pth')
        
    model = torch.load(SENTIMENT_MODEL_DISTIL)
    model.to('cuda')

    result['distil_sentiment_prediction_and_confidence_score'] = result.apply(lambda x: predict_sentiment_distil(x, tokenizer, model, return_both=True), axis=1)
    result['distil_sentiment_prediction'] = result['distil_sentiment_prediction_and_confidence_score'].apply(lambda x: x[0])
    result['distil_sentiment_confidence_score'] = result['distil_sentiment_prediction_and_confidence_score'].apply(lambda x: x[1])

    # Sentiment Prediction (Model 2)
    result['sentence3'] = result.apply(create_sentence_for_sentiment_seq2seq, axis=1)

    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_SEQ2SEQ)
    model = BartForConditionalGeneration.from_pretrained(SENTIMENT_MODEL_SEQ2SEQ)
    model.to('cuda')

    texts_to_predict = result.sentence3.tolist()
    if texts_to_predict:
        predicted_data = predict_sentiment_bart_model(texts_to_predict, model, tokenizer, batch_size=32, return_both=True)
    result['seq2seq_sentiment_prediction'] = pd.Series(predicted_data[0])
    result['seq2seq_sentiment_confidence_score'] = pd.Series(predicted_data[1])

    result['final_sentiment_prediction'] = result.apply(get_final_sentiment_pred, axis=1)

    result = result[original_columns + ['entity_ids', 'final_aspect_labels', 'entity_category', 'final_aspect_categories',  'final_sentiment_prediction']]
    
    return result