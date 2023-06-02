import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt



#loading the data and spreate the features from the target
def data_processing(file):
	column_names = ['ID', 'target'] + [f'feature#{i}' for i in range(30)]
	data = pd.read_csv(file, header=None, names=column_names)
	X = data.iloc[:, 2:].values
	data['target']=np.where(data["target"]=='M',1,0)
	y = data['target'].values
	return X,y
	

class MLPmodel:
	def __init__(self, input_size,hidden_size,output_size,learning_rate):
			np.random.seed(0)
			self.w1=np.random.rand(input_size,hidden_size)
			self.b1 = np.zeros((1, hidden_size))
			self.w2=np.random.rand(hidden_size,output_size)
			self.b2 = np.zeros((1, output_size))
			self.learning_rate=learning_rate
			self.acc_value=0.95


	#activation function
	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	#activation function derivative
	def sigmoid_derivative(self, x):
		return x * (1 - x)

	def forward(self, X):
		#calculate the SOP and pass to activation function
		 self.z1 = np.dot(X, self.w1) + self.b1
		 self.a1 = self.sigmoid(self.z1)
		 self.z2 = np.dot(self.a1, self.w2) + self.b2
		 self.a2 = self.sigmoid(self.z2)
		 return self.a2

	def backward(self, X, y, output):   
		#calculate the delta and update the whaights + baios
		d2 = (output - y) * self.sigmoid_derivative(output)
		d1 = np.dot(d2, self.w2.T) * self.sigmoid_derivative(self.a1)
		self.w2 -= self.learning_rate * np.dot(self.a1.T, d2)
		self.b2 -= self.learning_rate * np.sum(d2, axis=0, keepdims=True)
		self.w1 -= self.learning_rate * np.dot(X.T, d1)
		self.b1 -= self.learning_rate * np.sum(d1, axis=0, keepdims=True)

	def predict(self, X):
		#get the model prediction 
		return (self.forward(X) > 0.5).astype(int)

	#loss function
	def mean_squared_error(self, actual, predicted):
		sum_square_error = 0.0
		for i in range(len(actual)):
			sum_square_error += (actual[i] - predicted[i])**2.0
			mean_square_error = 1.0 / len(actual) * sum_square_error
		return mean_square_error

	#compute the accuricy
	def compute_acc(self, X, Y):
		correct=0
		for i in range(X.shape[0]):
			if X[i]==Y[i]:
				correct+=1
		acc=(correct/float(X.shape[0]))*100
		return acc

	def save_model(self,threshold):
		#save the model & update the accuracy value
		if(threshold>self.acc_value):
			self.acc_value=threshold
			joblib.dump(mlp, "Completed_model.joblib")

    
	def train(self, X, y, epochs,j):
		y = y.reshape(-1, 1)
		train_acc = []
		test_acc = []
		train_loss = []
		test_loss = []
		for i in range(epochs):

			#forward
			output=self.forward(X)

			#backword
			self.backward(X, y, output)

			#get this epoch predection
			pred_X_train=self.predict(X_train)
			pred_X_test=self.predict(X_test)

			#save teh model  accuracy history
			acc_tr = self.compute_acc(pred_X_train, y_train)
			train_acc.append(acc_tr)
			
			acc_ts = self.compute_acc(pred_X_test, y_test)
			test_acc.append(acc_ts)

			#save the model loss history
			train_ls=self.mean_squared_error(y_train, pred_X_train)
			train_loss.append(train_ls)

			test_ls=self.mean_squared_error(y_test, pred_X_test)
			test_loss.append(test_ls)

			#save a copy of the model 
			self.save_model(accuracy_score(y_test, pred_X_test))
	
			
		if(j==0):
			#ploting the  accuracy during the epochs
			plt.figure(1)
			plt.subplot(121)
			x=[i for i in range(epochs)]
			plt.title('trin/test accuracy')
			plt.plot(x,train_acc,label='train_accuracy')
			plt.plot(x,test_acc,label='test_accuracy')
			plt.legend()
	   
			#ploting the loss during the epochs
			plt.subplot(122)
			x=[i for i in range(epochs)]
			plt.title('train/test loss')
			plt.plot(x,train_loss,label='train_loss')
			plt.plot(x,test_loss,label='test_loss')
			plt.legend()
		
		
		#acuracy and train/test loss at each learning rate value
		if(j>0):
			plt.figure(2)
			print("Accuracy of the learning rate %.2f: "%(self.learning_rate), accuracy_score(y_test, pred_X_test))
			#plot larning rate graph
			plt.subplot(2,2,j)
			plt.title('Learning Rate: %.2f' % (self.learning_rate))
			plt.plot(train_loss,label='train_loss')
			plt.plot(test_loss,label='test_loss')
			plt.legend()
		



if __name__=="__main__":
	X,y=data_processing("wdbc.data")
	#spliting the data 
	X_train, X_test, y_train, y_test = train_test_split(X, y)
	# Standardize the features
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)
	#creat MLp model
	mlp = MLPmodel(30, 16, 1, 0.01)
	#start trinning
	mlp.train(X_train, y_train,200,0)
	#load and get the model accuricy
	loaded_model = joblib.load("Completed_model.joblib")
	y_predect=loaded_model.predict(X_test)
	print("final Accuracy: ", accuracy_score(y_test,y_predect),"\n")
	print(classification_report(y_predect,y_test))
	#Hyperparameters tuning
	learning_rates = [1.0, 0.5, 0.1, 0.01]
	hidden_sizes = [5, 10, 15, 20, 25, 30]
	test_accurcy=[]
	train_accurcy=[]
	#learning rate try
	for lr in range(len(learning_rates)):
		mlp = MLPmodel(30, 16, 1, learning_rates[lr])
		mlp.train(X_train, y_train,200,lr+1)
	
	
	#hidden size try
	for hs in range(len(hidden_sizes)):	
		mlp = MLPmodel(30, hidden_sizes[hs],1, 0.01)
		mlp.train(X_train, y_train,200,-1)
		test_accurcy.append(accuracy_score(y_test,mlp.predict(X_test)))
		train_accurcy.append(accuracy_score(y_train,mlp.predict(X_train)))
		print("Accuracy with %d nodes: "%(hidden_sizes[hs]), accuracy_score(y_test, mlp.predict(X_test))) 
	#plo the accurcy at each number of nodes
	plt.figure(3)
	plt.title('accurcy at each number of nodes')
	plt.plot(hidden_sizes,test_accurcy, label='test_accuracy')
	plt.plot(hidden_sizes,train_accurcy, label='train_accuracy')
	plt.ylabel('accurcy')
	plt.xlabel('hidden layer size')
	plt.legend()
	plt.show()
	