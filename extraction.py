# import requests
# import time
# import pandas as pd

# client_id = 'UZT7eG9q65gEwOc5IQgiKA'
# client_secret = 'EsaTm6IWzG9LCXn0-s2i9TLeV2agRg'
# username = 'username'
# password = 'password'
# user_agent = 'CommentExtractor:v2.0 (by u/username)'

# # Step 1: Authenticate and get access token
# auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
# data = {'grant_type': 'password', 'username': username, 'password': password}
# headers = {'User-Agent': user_agent}

# res = requests.post('https://www.reddit.com/api/v1/access_token', auth=auth, data=data, headers=headers)
# TOKEN = res.json()['access_token']
# headers['Authorization'] = f'bearer {TOKEN}'

# # List of subreddits to scrape
# subreddits = ['AskReddit', 'AmItheAsshole', 'worldnews', 'gaming', 'movies']

# # Number of comments to collect from each subreddit
# COMMENTS_PER_SUBREDDIT = 10000

# # Helper function to fetch comments
# def fetch_comments_from_subreddit(subreddit, target=COMMENTS_PER_SUBREDDIT):
#     print(f"\nCollecting comments from r/{subreddit}...")
#     all_comments = []
#     after = None
#     post_count = 1

#     while len(all_comments) < target:
#         params = {'limit': 50}
#         if after:
#             params['after'] = after

#         try:
#             posts_res = requests.get(f"https://oauth.reddit.com/r/{subreddit}/hot",
#                                      headers=headers, params=params)
#             posts = posts_res.json().get('data', {}).get('children', [])
#         except Exception as e:
#             pass
#             break

#         if not posts:
#             print("No more posts found.")
#             break

#         for post in posts:
#             post_id = post['data']['id']
#             post_title = post['data']['title']

#             # Fetch comments
#             try:
#                 comments_res = requests.get(f"https://oauth.reddit.com/comments/{post_id}",
#                                             headers=headers, params={'limit': 500})
#                 comments_json = comments_res.json()

#                 # Sometimes Reddit returns empty or malformed response
#                 if len(comments_json) < 2:
#                     continue

#                 comment_items = comments_json[1]['data']['children']
#                 for item in comment_items:
#                     if item['kind'] != 't1':
#                         continue
#                     body = item['data'].get('body')
#                     if body and body not in ['[removed]', '[deleted]']:
#                         all_comments.append({
#                             'subreddit': subreddit,
#                             'post_id': post_id,
#                             'post_title': post_title,
#                             'comment': body,
#                             'score': item['data'].get('score'),
#                             'author': item['data'].get('author'),
#                             'created_utc': item['data'].get('created_utc')
#                         })

#                 print(f"post-{post_count}: +{len(comment_items)} comments | Total: {len(all_comments)}")
#                 post_count += 1

#                 # Stop early if target reached
#                 if len(all_comments) >= target:
#                     break

#                 # Be nice to Reddit API
#                 time.sleep(2)

#             except Exception as e:
#                 pass
#                 time.sleep(1)
#                 continue

#         after = posts[-1]['data']['name']
#         time.sleep(0.1)

#     return all_comments


# # Step 2: Loop through subreddits and collect comments
# all_data = []

# for sub in subreddits:
#     comments = fetch_comments_from_subreddit(sub)
#     all_data.extend(comments)

# # Step 3: Save to CSV
# df = pd.DataFrame(all_data)
# df.to_csv("reddit_comments_multi_subs.csv", index=False)

# print(f"\nTotal comments collected: {len(df)}")