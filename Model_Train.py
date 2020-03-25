import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import librosa 
from sklearn.utils import shuffle #shuffle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)


csvData = pd.read_csv('JsonContentInCsvmultipleword.csv')
print(csvData.iloc[0, :])
batch_sz = 4 # batch size


file_names = []
labels = []


for i in range(0,len(csvData)):
    file_names.append(csvData.iloc[i, 0]) # First column is key which is path of audio file
    labels.append(csvData.iloc[i, 3]) # 3rd column (starting at 0) is label
file_names
1


def get_melspectrogram_db(file_path, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
  wav,sr = librosa.load(file_path,sr=sr)
  if wav.shape[0]<5*sr:
    wav=np.pad(wav,int(np.ceil((5*sr-wav.shape[0])/2)),mode='reflect')
  else:
    wav=wav[:5*sr]
  spec=librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft,
              hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
  spec_db=librosa.power_to_db(spec,top_db=top_db)
  return spec_db

def spec_to_image(spec, eps=1e-6):
  mean = spec.mean()
  std = spec.std()
  spec_norm = (spec - mean) / (std + eps)  # z-score normalization
  spec_min, spec_max = spec_norm.min(), spec_norm.max()
  spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min) # scaled to be between 0 and 255
  spec_scaled = spec_scaled.astype(np.uint8)
  return spec_scaled

def GetImageFromAudioFile(filepath):
  spec_db = get_melspectrogram_db(filepath)
  image = spec_to_image(spec_db)
  return image


class Model1(Dataset):
    def __init__(self, csv_path, file_path, rows_to_include):
        csvData = pd.read_csv(csv_path)
        csvData = shuffle(csvData)
        self.file_names = []
        self.labels = []
        
        
        for i in range(len(csvData)):
            if(i in rows_to_include):
                self.file_names.append(csvData.iloc[i, 0]) # First column is key which is path of audio file
                self.labels.append(csvData.iloc[i, 3]) # 3rd column (starting at 0) is label
        self.file_path = file_path
        
    def __getitem__(self, index):
        path = self.file_names[index]
                
        # Option 1, waveform
        #sound, sample_rate = torchaudio.load(path, out = None, normalization = True)
        #soundData = sound[0]        
        
        #option2 : mel-specgram
        #sound, sample_rate = torchaudio.load(path, out = None, normalization = True)
        #mel_specgram = torchaudio.transforms.Spectrogram()(sound)
        #soundData = mel_specgram
        #print(soundData.size())
        
        #option3: convert mel-specgram to image
        #sound, sample_rate = torchaudio.load(path, out = None, normalization = True)
        #mel_specgram = torchaudio.transforms.Spectrogram()(sound)
        #mel_specgram = spec_to_image(mel_specgram)
        #soundData = mel_specgram
        
        # Modify here to get different types of Audio processing.
        soundFormatted = GetImageFromAudioFile(path)
        soundFormatted = torch.from_numpy(soundFormatted)
        soundFormatted = soundFormatted.reshape(-1, 128, 157)
        return soundFormatted, self.labels[index]
    
    def __len__(self):
        return len(self.file_names)

#csv_path = 'JsonContentInCsv.csv'
#csv_path = 'JsonContentInCsvmultipleword.csv'
csv_path = 'JsonContentInCsvsingleword.csv'
file_path = ''

train_set = Model1(csv_path, file_path, range(1,5501))
test_set = Model1(csv_path, file_path, range(5502,6502))
print("Train set size: " + str(len(train_set)))
print("Test set size: " + str(len(test_set)))

kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {} #needed for using datasets on gpu
#kwargs = {'num_workers': 1, 'pin_memory': True}
print(device)


#train_loader = torch.utils.data.DataLoader(train_set, batch_size = 128, shuffle = True, **kwargs)
#test_loader = torch.utils.data.DataLoader(test_set, batch_size = 128, shuffle = True, **kwargs)

train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_sz, shuffle = True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_sz, shuffle = True, **kwargs)


# Model inspired from audio classifier tutorial of Pytorch (open in google colab) from 
# https://pytorch.org/tutorials/beginner/audio_classifier_tutorial.html?highlight=audio

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, padding_mode='VALID')
        self.bn1 = nn.BatchNorm2d(20)
        self.pool1 = nn.MaxPool2d(4)
        
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=5, padding_mode='VALID')
        self.bn2 = nn.BatchNorm2d(20)
        self.pool2 = nn.MaxPool2d(4)
        
        # Disable GRU, if you are using ReLU
        #self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        #self.fc = nn.Linear(hidden_dim, output_dim)
        #self.relu = nn.ReLU()
 
        self.lstm = nn.LSTM(input_size = 960, hidden_size =20, num_layers =1, batch_first=True, dropout=0.5)
        self.fc2 = nn.Linear(in_features = 20, out_features =20, bias = True)
        #self.relu = nn.ReLU() Put it here only if you want to learn its parameter
        
        
        # Dense bottleneck layer
        #self.linear = nn.Linear(hidden_size*6 + features.shape[1], lin_size)
        self.dense1 = nn.Linear(in_features = 20, out_features = 1, bias=True)
        self.dense2 = nn.Linear(in_features = 1, out_features = 1, bias=True)
        #self.tanh = torch.tanh()
        self.dropout = nn.Dropout(0.5) # Layer 8: A dropout layer applied to bottleneck layer
        
        
        # Layer 9: Dysarthric  dense layer --> Linear
        #self.dysarthriclayer = nn.Linear(1, 1)
        
        self.avgPool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(1, 1)
        
    def forward(self, x):
        #print('shape of input x: ', x.shape)
        x = x.permute(0,1,2,3)# Doesn't change anything.
        
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.dropout(x)
        X = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # GRU layer
        #print('shape1: ', x.shape)
        #h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) 
        #c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        #h0 = Variable(torch.zeros(1, x.size(0), 20)) 
        #c0 = Variable(torch.zeros(1, x.size(0), 20))
        x = x.reshape(1, 4, 48*20)
        #print('shape of input x0: ', x.shape)
        #x, _ = self.gru(x, (h0,c0))
        x, _ = self.lstm(x)
        x = self.dropout(x)
        #x = x[:, -1, :]
        x = x.reshape(4, 20)
        x = self.fc2(x)
        x = F.relu(x)
        
        #print('shape of input x1: ', x.shape)
        # 2 Dense layers
        x = self.dense1(x)
        x = self.dropout(x)
        x = torch.tanh(x)
        
        #print('shape of input x2: ', x.shape)
        x = self.dense2(x)
        x = self.dropout(x)
        x = torch.tanh(x)
        
        # Dysarthric layer
        #print('shape of input x3: ', x.shape)
        #x = self.dysarthriclayer(x)

        #print(x)
        #x = self.avgPool(x)
        #x = x.view(-1, 20*3*batch_sz)
        x = self.fc1(x)
        y_hat = F.log_softmax(x, dim = 0) 
        return y_hat

model = Net()
model.to(device) #Must send the model to gpu, else remaining code picks up stale version of old model which was earlier sent to gpu
#print(model)



#optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay = 0.0001)
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.09)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)

def train(model, epoch):
    #if torch.cuda.is_available():
    #    model.cuda()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device, dtype=torch.float32)
        target = target.to(device, dtype=torch.float32) # Note that target should be long, why? Investigate
        data = data.requires_grad_() #set requires_grad to True for training
        output = model(data)
        output = output.view(-1,1)
        #target = target.view(-1,1).squeeze_()
        target = target.view(-1,1)
        
        # old code
        #output = output.permute(1, 0, 2) #original output dimensions are batchSize x 1 x number_of_classes
        #print('shape of output after permute', output.shape)
        #loss = F.nll_loss(output[0], target) #the loss functions expects a batchSizex10 input
        
        #print(output.shape)
        #print(target.shape)
        #print(output)
        #print(target)
        
        '''
        #loss = F.nll_loss(output, target) #the loss functions expects a batchSizex10 input
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        '''
        
        pos_weight = torch.ones([1])  # All weights are equal to 1
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = criterion(output, target)  # -log(sigmoid(1.5))

        #loss_fn.backward()
        optimizer.step()
        if batch_idx % log_interval == 0: #print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))
def test(model, epoch):
    #if torch.cuda.is_available():
    #    model.cuda()
    model.eval()
    correct = 0
    for data, target in test_loader:
        data = data.to(device, dtype=torch.float32)
        target = target.to(device, dtype=torch.long)
        output = model(data)

        _, pred = torch.max(output, 1)
        correct += pred.eq(target).cpu().sum().item()
        
        print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
        
# Actual training of the model
log_interval = 20
#for epoch in range(1, 41):
for epoch in range(1, 41):
    if epoch == 31:
        print("First round of training complete. Setting learn rate to 0.001.")
    scheduler.step()
    train(model, epoch)
    test(model, epoch)