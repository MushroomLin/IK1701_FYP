
from keras.utils import plot_model
from keras.models import Sequential, load_model
model=load_model('./lstm_model')
plot_model(model, show_shapes=True,to_file='model.png')