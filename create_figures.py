import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

def data_summary(df):
    num_posts = df[df['type'] == 'post'].shape[0]
    num_comments = df[df['type'] == 'comment'].shape[0]

    unique_subreddits = df['subreddit'].nunique()

    df['date'] = pd.to_datetime(df['created_utc'], unit='s')
    date_min = df['date'].min().date()
    date_max = df['date'].max().date()

    avg_score_posts = df[df['type'] == 'post']['score'].mean()
    avg_score_comments = df[df['type'] == 'comment']['score'].mean()

    summary = pd.DataFrame({
        'Metric': ['Number of posts', 'Number of comments', 'Unique subreddits', 'Date range', 'Avg post upvotes', 'Avg comment upvotes'],
        'Value': [num_posts, num_comments, unique_subreddits, f"{date_min} to {date_max}", round(avg_score_posts,2), round(avg_score_comments,2)]
    })

    return summary

def show_wordcloud(df):
    positive_text = " ".join(df[df['sentiment_label']=='POSITIVE']['text'].dropna())
    negative_text = " ".join(df[df['sentiment_label']=='NEGATIVE']['text'].dropna())

    wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
    wordcloud_neg = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_text)

    plt.figure(figsize=(16,6))

    plt.subplot(1,2,1)
    plt.imshow(wordcloud_pos, interpolation='bilinear')
    plt.title('Positive Sentiment Word Cloud')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(wordcloud_neg, interpolation='bilinear')
    plt.title('Negative Sentiment Word Cloud')
    plt.axis('off')

    plt.show()

def plot_sentiment_distribution(df):
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x='sentiment_label', hue='type')
    plt.title("Sentiment Distribution by Post vs Comment")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()

    plt.show()

if __name__ == "__main__":

    df = pd.read_csv("reddit_sentiment_flat.csv")

    summary_df = data_summary(df)
    print(summary_df)

    show_wordcloud(df)
    
    plot_sentiment_distribution(df)