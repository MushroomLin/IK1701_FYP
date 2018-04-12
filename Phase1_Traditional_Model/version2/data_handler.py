from textblob import TextBlob
import re

def clean_headline(headline):
	# Remove links and special characters using regex
	return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", headline).split())

def analize_sentiment(headline):
	# Classify the polarity of a headline using textblob
	analysis = TextBlob(clean_headline(headline))
	if analysis.sentiment.polarity > 0:
		return 1
	elif analysis.sentiment.polarity == 0:
		return 0
	else:
		return -1

if __name__ == '__main__':
	sentence_list = ['how about you?',
	'holly shit',"nice shot",'what the fuck','excellent!']
	for sentence in sentence_list:
		print (sentence + ': ' + str(analize_sentiment(sentence)))