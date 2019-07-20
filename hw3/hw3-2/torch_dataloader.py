from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
	def __init__(self , train_x , train_y):
		self.train_x = train_x
		self.train_y = train_y

	
	def __getitem__(self, index):
		data = self.train_x[index]
		label = self.train_y[index]

		return data , label

	def __len__(self):
		return len(self.train_x)


      

