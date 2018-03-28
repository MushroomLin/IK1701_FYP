

predictions = model.predict(test_x)
money,real=stock_algorithm(predictions,test_y)
print(money,real)
x=range(0,len(test_y))
plt.plot(x,predictions, 'o', label="prediction")
plt.plot(x,test_y,'x',label='ground truth')
plt.show()
