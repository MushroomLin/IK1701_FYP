stock_lstm.py: script to run the LSTM model. 
	Output: 1. A file named 'AAPL_plot_x.csv' with the columns 'Date', 'Real' and 'Predict' indicating the date, real stock price and predicted price, respectively. 2. A file named 'loss.txt' recording the training and testing loss of different time spans.

	Run the script:
	$ python stock_lstm.py

plot.py: script to plot the figure of the predicted price (red) and the real values (blue) versus date.

	Run the script:
	$ python plot.py

plot_loss.py: script to plot the figure of the loss variation of the training (blue) and testing (red) process, as well as the baseline loss (green).

	Run the script:
	$ python plot_loss.py

config.py: script to record a set of parameters. Necessary explanations are given in comments after each parameter in the script.