import csv
import matplotlib.pyplot as plt
import config
FILE = config.COMPANY + '_plot_' + str(config.TIME_SPAN) + '.csv'
with open(FILE, 'rb') as csvfile:
    # get data length
    length = len(csvfile.readlines()) - 1 
    origin_length = length + 2*config.TIME_SPAN
    # determine start position for plotting
    if config.PLOT_TRAIN:
        start_pos = 0
    else:
        start_pos = int(origin_length * config.TRAIN_RATIO - config.TIME_SPAN)

with open(FILE, 'rb') as csvfile:
    # get plotting data from file
    cursor = -1
    date = []
    real = []
    predict = []
    reader = csv.reader(csvfile)
    for row in reader:
        if cursor < start_pos:
            cursor += 1
            continue
        else:
            date.append(row[0])
            real.append(float(row[1]))
            predict.append(float(row[2]))

date_num = [i for i in range(len(date))]
date_dict = dict(zip(date, date_num))

# Plot 
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)
plt.plot(date_num, real, 'bo', label='Real')
plt.plot(date_num, predict, 'ro', label='Predict')
plt.xlabel('Date')
plt.ylabel('Price')
xticks_num = []
xticks_label = []
for i in range(10):
    xticks_label.append(date[int(len(date)*(float(i)/10))][2:])
    xticks_num.append(date_dict[date[int(len(date)*(float(i)/10))]])
xticks_label.append(date[-1][2:])
xticks_num.append(date_dict[date[-1]])
yticks_num = []
max_value = max(max(real),max(predict))
min_value = min(min(real),min(predict))
lower_bound = int(min_value - (max_value - min_value) / 2)
upper_bound = int(max_value + (max_value - min_value) / 3)
for i in range(11):
    yticks_num.append(int(lower_bound + (upper_bound - lower_bound)*(float(i)/10)))
ax.set_xticks(xticks_num)
ax.set_yticks(yticks_num)
ax.set_xticklabels(xticks_label)
plt.xticks(rotation=45)
if config.TIME_SPAN > 1:
    plt.title('LSTM model with time span of ' + str(config.TIME_SPAN) + ' days')
else:
    plt.title('LSTM model with time span of 1 day')
plt.legend()
plt.show()