import os
from dotenv import load_dotenv
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import emoji
import unicodedata
from datetime import datetime, timedelta
import time

load_dotenv('./credentials/reddit.env')
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

def is_recent_video(published_date):
    one_week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat() + 'Z'
    return published_date >= one_week_ago

def search_videos(query, api_key, max_results=50, required_videos=5):
    youtube = build('youtube', 'v3', developerKey=api_key)
    videos = []
    next_page_token = None

    while len(videos) < required_videos:
        search_response = youtube.search().list(
            q=query,
            part='snippet',
            maxResults=max_results,
            type='video',
            order='relevance',
            pageToken=next_page_token
        ).execute()

        for item in search_response['items']:
            video_id = item['id']['videoId']
            title = item['snippet']['title']
            published_date = item['snippet']['publishedAt']
            if is_recent_video(published_date):
                videos.append({
                    'video_id': video_id,
                    'title': title,
                    'published_date': published_date
                })
                if len(videos) >= required_videos:
                    break

        next_page_token = search_response.get('nextPageToken')
        if not next_page_token and len(videos) < required_videos:
            break

    return videos

# Function to retrieve comments
def get_comments(video_id, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []

    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText"
        )
        response = request.execute()

        while request:
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                parent_comment_id = item['snippet']['topLevelComment']['id']
                parent_comment = comment['textDisplay']
                comment_date = comment['publishedAt']
                comments.append({
                    'video_id': video_id,
                    'video_title': None,  # Will be set later
                    'comment': parent_comment,
                    'comment_id': parent_comment_id,
                    'number_of_likes': comment['likeCount'],
                    'parent_comment': "N/A",
                    'parent_comment_id': "N/A",
                    'comment_date': comment_date
                })

                # Check if there are replies
                if item['snippet']['totalReplyCount'] > 0:
                    replies_request = youtube.comments().list(
                        part="snippet",
                        parentId=parent_comment_id,
                        maxResults=100,
                        textFormat="plainText"
                    )
                    replies_response = replies_request.execute()

                    for reply in replies_response['items']:
                        reply_comment = reply['snippet']['textDisplay']
                        reply_comment_id = reply['id']
                        reply_date = reply['snippet']['publishedAt']
                        comments.append({
                            'video_id': video_id,
                            'video_title': None,  # Will be set later
                            'comment': reply_comment,
                            'comment_id': reply_comment_id,
                            'number_of_likes': reply['snippet']['likeCount'],
                            'parent_comment': parent_comment,
                            'parent_comment_id': parent_comment_id,
                            'comment_date': reply_date
                        })

            if 'nextPageToken' in response:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    pageToken=response['nextPageToken'],
                    maxResults=100,
                    textFormat="plainText"
                )
                response = request.execute()
            else:
                break
    except HttpError as e:
        if e.resp.status == 403:
            if "commentsDisabled" in str(e):
                print(f"Comments are disabled for video {video_id}. Skipping...")
            elif "quotaExceeded" in str(e):
                print("Quota exceeded. Please try again later.")
                time.sleep(600)  # Sleep for 10 minutes before retrying
        else:
            raise e

    return comments


def scrape_youtube(save_path_and_file_name, videos_to_scrape=5):
    search_queries = ["Trump", "Kamala"]
    all_comments = []

    # Load the existing CSV
    try:
        existing_df = pd.read_csv(save_path_and_file_name)
    except FileNotFoundError:
        existing_df = pd.DataFrame()

    for query in search_queries:
        valid_videos = []
        while len(valid_videos) < videos_to_scrape:
            videos = search_videos(query, YOUTUBE_API_KEY, max_results=50, required_videos=videos_to_scrape+5)
            for video in videos:
                video_id = video['video_id']
                published_date = video['published_date']
                if not is_recent_video(published_date):
                    continue

                comments = get_comments(video_id, YOUTUBE_API_KEY)
                if comments:
                    video_title = video['title']

                    # If video ID exists in the existing CSV
                    if 'video_id' in existing_df.columns and video_id in existing_df['video_id'].values:
                        existing_comments = existing_df[existing_df['video_id'] == video_id]['comment_id'].values
                        for comment in comments:
                            if comment['comment_id'] not in existing_comments:
                                comment['video_title'] = video_title
                                comment['video_post_date'] = published_date
                                all_comments.append(comment)
                    else:
                        # If video ID is new
                        for comment in comments:
                            comment['video_title'] = video_title
                            comment['video_post_date'] = published_date
                            all_comments.append(comment)

                    valid_videos.append(video)
                if len(valid_videos) >= videos_to_scrape:
                    break

    # Convert to DataFrame
    new_df = pd.DataFrame(all_comments, columns=['video_title', 'video_id', 'comment', 'comment_id', 'number_of_likes', 'parent_comment', 'parent_comment_id', 'comment_date', 'video_post_date'])

    # Append new comments to existing CSV
    if not existing_df.empty:
        updated_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['comment_id'], keep='first')
    else:
        updated_df = new_df

    # Save updated DataFrame to CSV
    updated_df.to_csv(save_path_and_file_name, index=False)
    print(f"YouTube comments have been saved to {save_path_and_file_name}")

    return updated_df