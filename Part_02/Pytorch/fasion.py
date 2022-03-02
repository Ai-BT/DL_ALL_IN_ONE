# %%

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tr

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,

    # torch tensor 로 바꿔주는 거 (/255)
    # 0 ~ 1 사이 값으로 바꿔준다
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
# %%

# 데이터 시각화하기
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# %%

# DataLoader 만들기
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

# %%

# DataLoader를 통해 반복하기(iterate)
# 이미지와 정답(label)을 표시합니다.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

# matplotlib 에서는 gray scale에서 채널이 없어야 보여준다.
# 그래서 squeeze 진행한다.
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
# %%

# 간단한 Custom Dataset/Transform/DataLoader 만들기
# 나의 데이터로 학습

class CustomDataset(Dataset):
  def __init__(self, np_data, transform=None):
    self.data = np_data
    self.transform = transform
    self.len = np_data.shape[0] # 데이터 수

  def __len__(self):
    return self.len

  def __getitem__(self, idx):    
    sample = self.data[idx]
    if self.transform:
      sample = self.transform(sample)
    return sample


# %%

# 간단한 함수
def square(sample):
  return sample**2
# %%

# 함수를 transforms 
# 여러개 함수 가능
trans = tr.Compose([square])
# %%

# 0 ~ 9 까지 데이터
np_data = np.arange(10)
custom_dataset = CustomDataset(np_data, transform=trans)

# %%

# data 설정
custom_dataloader = DataLoader(custom_dataset, batch_size=2, shuffle=True)
# %%

# 학습 진행
for _ in range(3):
  for data in custom_dataloader:
    print(data)
  print("="*20)


# %%

# Model 

# device 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
# %%

# Model 만들기

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 128), # tensor 에서 dense 같은 놈
            nn.ReLU(),
            nn.Dropout(0.2),            
            nn.Linear(128, 10)
            # 소프트맥스는 나중에 알아서 해줘서 뺀거다
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# %%

# Model instance 생성, device 설정
model = NeuralNetwork().to(device)
print(model)
# %%

# 가상의 data 만들어서 예측해보기
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits) # 소프트 맥스
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
# %%

# 손실 함수를 초기화합니다.
loss_fn = nn.CrossEntropyLoss()
# %%

# Optimizer
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# %%

# Training을 위한 함수
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # 예측(prediction)과 손실(loss) 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Test를 위한 함수
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
# %%

# 학습 진행하기
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
# %%

# Model 저장하고 불러오기

# 학습된 model parameter 저장
torch.save(model.state_dict(), 'model_weights.pth')


# %%

# 새 Model instance 생성, device 설정
model2 = NeuralNetwork().to(device)
print(model2)

# %%

# test
model2.eval()
test_loop(test_dataloader, model2, loss_fn)
# %%

# 저장한 parameter 불러오기
model2.load_state_dict(torch.load('model_weights.pth'))

# %%

# test
model2.eval()
test_loop(test_dataloader, model2, loss_fn)

# %%
