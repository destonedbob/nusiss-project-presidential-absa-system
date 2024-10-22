from transformers import pipeline
from dotenv import load_dotenv
import os
import praw
import datetime
import pandas as pd
import time
from utilities.util import create_folder_if_not_exists

load_dotenv('./credentials/reddit.env')
APP_NAME = os.getenv('APP_NAME')
SECRET_KEY = os.getenv('SECRET_KEY')
APP_ID = os.getenv('APP_ID')
USERNAME = os.getenv('REDDIT_USERNAME')
USER_AGENT = f'{APP_NAME} by u/{USERNAME}'

def get_reddit_object(client_id=APP_ID, client_secret=SECRET_KEY, user_agent=USER_AGENT, user_name=USERNAME):
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        user_name=user_name
        )
    return reddit

def get_qa_object():
    qa_obj = pipeline(model="facebook/bart-large-mnli")
    return qa_obj

def get_trump_reddit_posts_and_comments(reddit, qa_obj, save_dir, max_posts_to_collect=10):
    num_collected_posts = 0
    post_data_list = []
    comment_data_list = []

    search_term = "Trump"

    try: 
        posts = reddit.subreddit('all').search(search_term, limit=max_posts_to_collect+10, sort='top', time_filter='month')
    except:
        time.sleep(60*3)
        posts = reddit.subreddit('all').search(search_term, limit=max_posts_to_collect+10, sort='top', time_filter='month')

    for post in posts:
        
        if num_collected_posts >= max_posts_to_collect:
            break 

        answers = qa_obj(
            post.title,
            candidate_labels=["About Donald Trump", "About Donald Trump's family", "Not about related to Donald Trump"],
        )

        index_of_related = answers['labels'].index("About Donald Trump")
        related_probability = answers['scores'][index_of_related]
        
        if related_probability < 0.5:
            continue
        
        post_data = {
            'post_id': post.id,
            'post_title': post.title,
            'votes': post.score,
            'created_utc': datetime.datetime.utcfromtimestamp(post.created).strftime('%d-%b-%Y'),
            'num_comments': post.num_comments,
            'post_body': post.selftext if post.is_self else post.url
        }
        
        post_data_list.append(post_data)

        # Fetch comments
        try: 
            post.comments.replace_more(limit=None)  # Ensure all comments are fetched
        except:
            time.sleep(60*3)
            post.comments.replace_more(limit=None)  # Ensure all comments are fetched


        try: 
            comment_list = post.comments.list()
        except:
            time.sleep(60*3)
            comment_list = post.comments.list()

        for comment in comment_list:

            # Print or save the scraped information
            comment_data = {
                'Post Title': post.title,
                'Post ID': post.id,
                'Comment ID': comment.id,
                'Timestamp': datetime.datetime.utcfromtimestamp(comment.created).strftime('%d-%b-%Y'),
                'Votes': comment.score,
                'Comment': comment.body,
                'Parent Comment ID': comment.parent_id.split('_')[-1] if 't1_' in comment.parent_id else None
            }

            comment_data_list.append(comment_data)
        
        num_collected_posts += 1

    post_df = pd.DataFrame(post_data_list)
    comment_df = pd.DataFrame(comment_data_list)

    create_folder_if_not_exists(save_dir)

    post_df.to_csv(f'{save_dir}{datetime.datetime.today().strftime("%d%m%Y")}_reddit_post_data_trump.csv', index=False)
    comment_df.to_csv(f'{save_dir}{datetime.datetime.today().strftime("%d%m%Y")}_reddit_comment_data_trump.csv', index=False)

    return post_df, comment_df

def get_kamala_reddit_posts_and_comments(reddit, qa_obj, save_dir, max_posts_to_collect=10):
    num_collected_posts = 0
    post_data_list = []
    comment_data_list = []

    search_term = "Kamala"

    try: 
        posts = reddit.subreddit('all').search(search_term, limit=max_posts_to_collect+10, sort='top', time_filter='month')
    except:
        time.sleep(60*3)
        try:
            posts = reddit.subreddit('all').search(search_term, limit=max_posts_to_collect+10, sort='top', time_filter='month')
        except:
            time.sleep(60)


    for post in posts:
        
        if num_collected_posts >= max_posts_to_collect:
            break 

        answers = qa_obj(
            post.title,
            candidate_labels=["Kamala Harris", "Not Kamala Harris"],
        )

        index_of_related = answers['labels'].index("Kamala Harris")
        related_probability = answers['scores'][index_of_related]
        
        if related_probability < 0.5:
            continue
        
        post_data = {
            'post_id': post.id,
            'post_title': post.title,
            'votes': post.score,
            'created_utc': datetime.datetime.utcfromtimestamp(post.created).strftime('%d-%b-%Y'),
            'num_comments': post.num_comments,
            'post_body': post.selftext if post.is_self else post.url
        }
        
        post_data_list.append(post_data)

        # Fetch comments
        try: 
            post.comments.replace_more(limit=None)  # Ensure all comments are fetched
        except:
            time.sleep(60*3)
            try:
                post.comments.replace_more(limit=None)  # Ensure all comments are fetched
            except:
                time.sleep(60)


        try: 
            comment_list = post.comments.list()
        except:
            time.sleep(60*3)
            try:
                comment_list = post.comments.list()
            except:
                time.sleep(60)

        for comment in comment_list:

            # Print or save the scraped information
            comment_data = {
                'Post Title': post.title,
                'Post ID': post.id,
                'Comment ID': comment.id,
                'Timestamp': datetime.datetime.utcfromtimestamp(comment.created).strftime('%d-%b-%Y'),
                'Votes': comment.score,
                'Comment': comment.body,
                'Parent Comment ID': comment.parent_id.split('_')[-1] if 't1_' in comment.parent_id else None
            }

            comment_data_list.append(comment_data)
        
        num_collected_posts += 1

    post_df = pd.DataFrame(post_data_list)
    comment_df = pd.DataFrame(comment_data_list)

    create_folder_if_not_exists(save_dir)

    post_df.to_csv(f'{save_dir}{datetime.datetime.today().strftime("%d%m%Y")}_reddit_post_data_kamala.csv', index=False)
    comment_df.to_csv(f'{save_dir}{datetime.datetime.today().strftime("%d%m%Y")}_reddit_comment_data_kamala.csv', index=False)

    return post_df, comment_df

def read_paths_create_df(paths):
    result = pd.DataFrame()
    first = True
    for path in paths:            
        temp_df = pd.read_csv(path)
        if 'trump' in path:
            temp_df['search_type'] = 'Trump'
        else:
            temp_df['search_type'] = 'Kamala'
        if first:
            first_columns = list(temp_df.columns)
        assert first_columns == list(temp_df.columns)
        result = pd.concat([result, temp_df])
        first = False

    return result 