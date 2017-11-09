import got3 as got
stocks=['AMZN','GOOGL','AAPL','MSFT','NFLX']

def printTweet(descr, t):
    print(descr)
    print("Favorite: %s" % t.favorites)
    print("Retweets: %d" % t.retweets)
    print("Text: %s" % t.text)
    print("Date: %s" % t.date)
    print("Hashtags: %s\n" % t.hashtags)
def show(query, username, date, maxTweets):
    tweetCriteria = got.manager.TweetCriteria()
    if query:
        tweetCriteria.setQuerySearch(query)
    if username:
        tweetCriteria.setUsername(username)
    if date:
        tweetCriteria.setSince(date[0]).setUntil(date[1])
    if maxTweets:
        tweetCriteria.setMaxTweets(maxTweets)
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    for tweet in tweets:
        printTweet("Twitter:", tweet)
# Get tweets by query search
show('#google',None,['2016-01-01','2016-04-02'],10)