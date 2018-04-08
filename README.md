# IK1701_FYP: Stock Market Prediction with Social Media and News Data through a Deep Learning Model
## Overall Dependency
* Data Preprocess:  
	NLTK: http://www.nltk.org
* Machine learning:  
    Scikit-learn: http://scikit-learn.org/  
    Tensorflow: https://www.tensorflow.org/  
    Keras: https://keras.io  
* Matrix Computation:  
    numpy: http://www.numpy.org  
    pandas: https://pandas.pydata.org
* Data Visualize:  
    matplotlib: https://matplotlib.org
## Phase I: Using Kaggle data for stock prediction with traditional machine learning model
### Data
* Data Kaggle:  
	Daily News for Stock Market Prediction  
	https://www.kaggle.com/aaron7sun/stocknews
* Sentiment Analysis:  
	OpinionFinder: http://mpqa.cs.pitt.edu/opinionfinder/  
	SentiWordNet: http://sentiwordnet.isti.cnr.it  

### File Description
#### Version 1:
* sentiwordnet/sentiwordnet.tsv: TSV file of sentiwordnet
* data_helper.py: File for data cleaning and preprocess
* sentiwordnet.py: Class file for build sentiwordnet
* train.py: Main file for train the model
#### Version 2:
A voting model
* data_handler.py: File for data cleaning and preprocess
* train.py: Main file for train the model
### Usage: 
    python train.py
    
## Phase II: Using Kaggle data for stock prediction with Convolutional Neural Network
### Data
* Data Kaggle:
	Daily News for Stock Market Prediction
	https://www.kaggle.com/aaron7sun/stocknews
### File Description
* cnn.py: Class for build the CNN model
* cnn_train.py: Main file for train the model
* data_function.py: File for data cleaning and preprocess
* eval.py: File for test the result and make prediction
* word2vec_model, word2vec_model.syn1neg.npy, word2vec_model.wv.syn0.npy:  
Word2Vec file for word embedding, too large to uploaded. You can run code without this
### Usuage
    # Train the model
    python cnn_train.py --parameter=value
    * Data Loading Parameters  
    --dev_sample_percentage: Percentage of the training data to use for validation  
    --positive_data_file: Data source for the positive data.  
    --negative_data_file: Data source for the negative data.  
    * Model Hyperparameters  
    --embedding_dim: Dimensionality of character embedding (default: 128)  
    --filter_sizes: Comma-separated filter sizes (default: '3,4,5')  
    --word2vec: word2vec embedding file  
    --num_filters: Number of filters per filter size (default: 128)  
    --dropout_keep_prob: Dropout keep probability (default: 0.5)  
    --l2_reg_lambda: L2 regularization lambda (default: 0.005)  
    * Training Parameters  
    --batch_size: Batch Size (default: 64)  
    --num_epochs: Number of training epochs (default: 200)  
    --evaluate_every: Evaluate model on dev set after this many steps (default: 100)  
    --checkpoint_every: Save model after this many steps (default: 100)  
    --num_checkpoints: Number of checkpoints to store (default: 5)  
    * Misc Parameters  
    --allow_soft_placement: Allow device soft device placement  
    --log_device_placement: Log placement of ops on devices
      
    # Test the model
    python eval.py --paarameter=value
    # Data Parameters
    --data_file: Data source for the test data.
    # Eval Parameters
    --batch_size: Batch Size (default: 64)
    --checkpoint_dir: Checkpoint directory from training run
    --eval_train: Evaluate on all training data
    * Misc Parameters  
    --allow_soft_placement: Allow device soft device placement  
    --log_device_placement: Log placement of ops on devices

## Phase III: Collect data from Twitter, Google finance and Yahoo finance
### Tools
* scrapy: https://scrapy.org
* getOldTweets: https://github.com/Jefferson-Henrique/GetOldTweets-python
* quandlAPI: https://www.quandl.com
### File description
* twitter_crawler: Crawler for twitter data
* yhscraper_google: Crawler for news data from google finance
* yhscraper_yahoo: Crawler for news data from yahoo finance
* price_data.py: Script for getting stock price data
### Usage
* See Readme file under each folder and document on quandl website

## Phase IV: Sentiment CNN training with existing big-scale dataset
### Data
* Sentiment 140: http://www.sentiment140.com
### File description
* cnn.py: Class for build the CNN model
* cnn_train.py: Main file for train the model
* data_function.py: File for data cleaning and preprocess
* eval.py: File for test the result and make prediction
* word2vec_model, word2vec_model.syn1neg.npy, word2vec_model.wv.syn0.npy:  
Word2Vec file for word embedding, too large to uploaded. You can run code without this
* /data/Twitter.csv: CSV file of Sentiment140 dataset, not upload. 
You can download by yourself
* /data/process_data.py: process data of Sentiment140, generate positive file and negative file
### Usage
    # prepare dataset
    Download sentiment140 dataset and rename as Twitter.csv 
    Put Twitter.csv under /data folder
    Enter /data folder
    python process_data.py
    # Train the model
    python cnn_train.py --parameter=value
    * Data Loading Parameters  
    --dev_sample_percentage: Percentage of the training data to use for validation  
    --positive_data_file: Data source for the positive data.  
    --negative_data_file: Data source for the negative data.  
    * Model Hyperparameters  
    --embedding_dim: Dimensionality of character embedding (default: 128)  
    --filter_sizes: Comma-separated filter sizes (default: '3,4,5')  
    --word2vec: word2vec embedding file  
    --num_filters: Number of filters per filter size (default: 128)  
    --dropout_keep_prob: Dropout keep probability (default: 0.5)  
    --l2_reg_lambda: L2 regularization lambda (default: 0.005)  
    * Training Parameters  
    --batch_size: Batch Size (default: 64)  
    --num_epochs: Number of training epochs (default: 200)  
    --evaluate_every: Evaluate model on dev set after this many steps (default: 100)  
    --checkpoint_every: Save model after this many steps (default: 100)  
    --num_checkpoints: Number of checkpoints to store (default: 5)  
    * Misc Parameters  
    --allow_soft_placement: Allow device soft device placement  
    --log_device_placement: Log placement of ops on devices
      
    # Test the model
    python eval.py --paarameter=value
    # Data Parameters
    --data_file: Data source for the test data.
    # Eval Parameters
    --batch_size: Batch Size (default: 64)
    --checkpoint_dir: Checkpoint directory from training run
    --eval_train: Evaluate on all training data
    * Misc Parameters  
    --allow_soft_placement: Allow device soft device placement  
    --log_device_placement: Log placement of ops on devices
## Phase V: Further data collection and development of sentiment CNN
### File description
Since the Factiva have anti-crawl, it is a stupid way to implement the crawller
You may need to change code
* auto_script.py: Example script for download rtf data
* rft_txt_converter.py: Example script for change rtf data to txt file
* txt_csv_converter.py: Script for change txt file into csv file
* /factiva_data/Apple_News_Data.csv: CSV file of data we collected of Apple Inc.
* /factiva_data/prediction.csv: Prediction result with CNN model
* /factiva_data/sentiment.csv: CSV file of stock price and sentiment of each day
* /factiva_data/extreme_days.py: Script for generate sentiment.csv
* /factiva_data/graph.py: Create graph of each day price and sentiment
* /factiva_data/explore_data.py: Example of how to use sentiment.csv
* /factiva_data/filter.py: Extract the news in the dates whose [argv] is in the top 10%  
    [argv] is the index speficied by user, which can be:  
    "Total": total number of news  
    "Average": proportion of positive news  
    "Change": changing rate of close price  
### Usage
    Since the Factiva have anti-crawl, it is a stupid way to implement the crawler
    You may need to change the code to run.
    python auto_script.py
    python rtf_txt_converter.py
    python txt_csv_converter.py
    python eval_data.py --checkpoint_dir=your_path_to_checkpoint
    python explore_data.py
    python extreme_days.py
    python graph.py
    python filter.py [argv]
## Phase VI: Development of an LSTM model for price prediction
### File description
* stock_lstm.py: script to run the LSTM model
* plot.py: script to plot the figure of the predicted price (red) and the real values (blue) versus date
* plot_loss.py: script to plot the figure of the loss variation of the training (blue) and testing (red) process, as well as the baseline loss (green)
* config.py: script to record a set of parameters. Necessary explanations are given in comments after each parameter in the script
### Usage
    python stock_lstm.py
    python plot.py
    python plot_loss.py
## Phase VII: Development of an LSTM model for price change prediction and an investment simulation algorithm
### File description
* create_data.py: create data for LSTM training
* lstm.py: LSTM model and algorithm visualize
### Usage
    # Create data
    python create_data.py [stock_symbol]
    # Train LSTM model
    python lstm.py
    * Configuration of LSTM
    instrument = 'AAPL'
    start_date = '2008-01-01'
    split_date = '2016-01-01'
    end_date = '2018-01-01'
    fields = [ 'change','adj_volume']  #features
    seq_len = 10 #length of input
    batch = 128 # batch size
    epochs = 100 # num of training epochs