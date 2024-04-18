## 문서의 목적

이 문서는 인공지능 학습을 위한 샘플을 제공하고 PyTorch 샘플의 사용법과 핵심적인 단계를 설명합니다. 이를 통해 PyTorch를 활용하여 인공지능 모델을 설계하고 학습시키는 방법을 설명합니다.

PyTorch는 Tensorflow와 비교하여, 동적 그래프를 기반으로 하여 모델을 정의하고 실행하는 과정을 단순화 시켜줍니다. 또한 복잡한 신경망 구조를 정의하고 수정하는 데 있어 보다 유연한 환경을 제공합니다. 이는 PyTorch를 이용하여 빠르게 신경망을 만들어 볼 수 있음을 의미합니다.

<br>

아래 영상은 샘플을 실행하여 테스트 하는 과정을 보여줍니다. 여기서 학습된 MNIST 이미지의 경우 상대적으로 정확하지만 학습되지 않은 이미지를 이용할 경우 전혀 다른 답이 나오는 것을 볼 수 있습니다.

<br>

https://github.com/kbmhansungb/kbmhansungb.github.io/assets/56149613/05c81c33-ea79-4e67-9c95-fd90264848d8

## PyTorch 설치

Anaconda를 이용해 PyTorch를 설치하는 방법을 설명합니다. PyTorch를 설치할 때 Python도 같이 설치되며 구체적인 설치 방법은 링크를 클릭하여 볼 수 있습니다.

<br>

1. 환경과 패키지 설치를 위한 [Anaconda](https://docs.anaconda.com/free/anaconda/install/windows/)를 설치합니다.
2. Anaconda Prompt를 이용하여 [PyTorch](https://pytorch.org/)를 설치합니다.

<br>

이제 인터프리터를 Anaconda의 Python으로 설정합니다. 여기서는 VSC를 이용합니다.

<br>

1. 'Ctrl + Shift + P'를 누르고 'Python: Select Interpreter' 커맨드를 입력합니다.
2. PyTorch를 설치한 Anaconda 환경으로 설정합니다.

<br>

## PyTorch를 이용한 인공지능 학습

인공지능 학습 과정은 데이터를 준비, 모델 설계, 학습과 평가의 단계로 구분할 수 있습니다. 인공지능을 학습시키기 위해 불러온 데이터를 모델에 입력하여 출력값이 기대값을 출력하도록 **역전파 알고리즘**으로 학습시킵니다.

<br>

이 문서에서 PyTorch에서 인공지능을 학습 시키는 예제의 구성요소는 다음과 같습니다:

### [DataSet](https://tutorials.pytorch.kr/beginner/basics/data_tutorial.html)

* Dataset: PyTorch에서 제공하는 데이터 셋 클래스로 데이터의 집합을 나타내고, 데이터에 간단한 전처리를 적용할 수 있습니다.
* DataLoader: 데이터셋을 반복 가능한 객체로 감싸는 역할을 하며, 배치 크기(batch size)를 설정하거나 데이터를 섞는(shuffle) 등의 기능을 제공하여 모델 학습 시 데이터를 효율적으로 로딩할 수 있도록 도와줍니다.

### [Transform](https://tutorials.pytorch.kr/beginner/basics/transforms_tutorial.html)

* Transformer: 데이터에 적용되는 전처리 과정으로, 데이터를 모델 학습에 적합한 형태로 변경합니다.

### [Define Model](https://tutorials.pytorch.kr/beginner/basics/buildmodel_tutorial.html)

* Model: 신경망의 전체 구조를 의미하며, 이 구조는 여러 계층(Layer)과 모듈(module)로 구성됩니다.
* Module: 신경망 모델의 기본 구성 요소로, 여러 연산들 또는 하위 모듈들의 집합 입니다.
* Layer: 신경망의 기본적인 계층으로, 특정 수학적 연산을 처리합니다.

### [Train & Test Model](https://tutorials.pytorch.kr/beginner/basics/optimization_tutorial.html)

* Loss Function: 모델의 예측 성능을 평가하기 위해 사용되는 함수로, 실제 값과 예측 값 사이의 차이를 계산합니다.
* Optimizer: 학습 과정에서 모델의 매개변수를 업데이트하는 방법을 정의합니다.

Environment

* Device: 모델이나 데이터를 처리할 하드웨어를 지정합니다.

<br>

## PyTorch 예제

예제에서 PyTorch 샘플 코드를 설명합니다. 이 샘플 코드는 모델을 정의하는 파이썬 코드와 학습, 테스트, 이를 로컬 웹으로 실행하는 파이썬 코드가 들어 있으며 웹을 위한 파일들과 이미지가 들어있습니다.

[Sample.zip](https://github.com/kbmhansungb/kbmhansungb.github.io/files/15020583/Sample.zip)

### 데이터 및 학습을 위한 폴더 구조

실행을 위한 파이썬 코드는 Sample 폴더 안에 있으며 모델을 정의한 코드와 모델에 대한 데이터는 모델 폴더 안에 있습니다. templates는 웹 페이지 html입니다. 또한 Sample 폴더 안에 테스트를 위한 이미지들 MNIST_Number, Other_Number가 있습니다.

<br>

![image](https://github.com/kbmhansungb/kbmhansungb.github.io/assets/56149613/1d66e9dd-a3af-49af-8c2c-56c9bc1d4155)

<br>

### 학습을 위한 Python 구성

필요한 모듈을 임포트 합니다.

```python
import torch
import torch.nn as nn
```

<br>

모델을 정의하고 사용하기 위한 설정을 합니다. 여기서 torch.device는 GPU를 이용한 연산을 지원하면 GPU를 사용하고 아니라면 CPU를 사용하도록 설정합니다. savedPath는 모델을 저장하고 불러올 때 사용할 경로입니다.

```python
# 환경설정을 정의합니다.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
savedPath = 'Model/Saved/model.ckpt'
```

<br>

신경망 모델을 정의합니다. `nn.Module`을 상속받아 NeuralNet을 만듭니다. NeuralNet은 Linear, ReLU, Linear 레이어를 가집니다. 이 후 모델을 생성하고 설정된 디바이스에서 처리하도록 합니다.

```python
# 신경망 모델을 정의합니다.
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
model = NeuralNet(784, 500, 10).to(device)
```

<br>

모델을 저장하고 불러오는 함수를 추가합니다.

```python
# 모델을 저장하고 불러오는 함수를 정의합니다.
def save_model():
    torch.save(model.state_dict(), savedPath)
    
def load_model():
    model.load_state_dict(torch.load(savedPath))
    model.eval()
```

<br>

`model.py`의 전체 코드는 다음과 같습니다:

```python
import torch
import torch.nn as nn

# 환경설정을 정의합니다.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
savedPath = 'Model/Saved/model.ckpt'

# 신경망 모델을 정의합니다.
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
model = NeuralNet(784, 500, 10).to(device)

# 모델을 저장하고 불러오는 함수를 정의합니다.
def save_model():
    torch.save(model.state_dict(), savedPath)
    
def load_model():
    model.load_state_dict(torch.load(savedPath))
    model.eval()
```

<br>

### 모델 학습시키기

정의한 모델을 포함하여 필요한 모듈을 임포트 합니다.

```python
import torch
import torch.nn as nn
import torchvision

import Model.model as model
```

<br>

학습을 위해 데이터셋과 로더를 설정합니다.

```python
# 학습을 위한 데이터셋을 불러옵니다.
train_dataset = torchvision.datasets.MNIST(root='../../data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
```

<br>

손실 함수와 옵티마이저를 생성하고, 학습을 몇 번 할지(Epochs)를 설정합니다.

```python
# 모델을 학습합니다.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.model.parameters(), lr=0.001)
total_step = len(train_loader)
num_epochs = 5
```

<br>

학습을 진행합니다. 먼저 num_epochs만큼 전체 데이터를 반복하여 학습 시킵니다.

train_loader에서 데이터를 읽어오면 device로 보내 계산합니다.

```python
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Device로 데이터를 보냅니다.
        images = images.to(model.device)
        labels = labels.to(model.device)
        
        images = images.reshape(-1, 28*28)
        
        # Forward pass
        outputs = model.model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# 모델을 저장합니다.
model.save_model()
```

<br>

`Train.py`의 전체 코드는 다음과 같습니다:

```python
import torch
import torch.nn as nn
import torchvision

import Model.model as model

# 학습을 위한 데이터셋을 불러옵니다.
train_dataset = torchvision.datasets.MNIST(root='../../data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

# 모델을 학습합니다.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.model.parameters(), lr=0.001)
total_step = len(train_loader)
num_epochs = 5

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Device로 데이터를 보냅니다.
        images = images.to(model.device)
        labels = labels.to(model.device)
        
        images = images.reshape(-1, 28*28)
        
        # Forward pass
        outputs = model.model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# 모델을 저장합니다.
model.save_model()
```

<br>

### 서비스를 위한 Python 구성

`Run.py`의 전체 코드는 다음과 같습니다:

```python
from flask import Flask, render_template, request, redirect, url_for
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

import Model.model as model

app = Flask(__name__)

# 모델 로드 (예시로 간단한 모델을 사용합니다)
model.load_model()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            image = Image.open(io.BytesIO(file.read()))
            result = predict(image)
            return render_template('result.html', result=result)
    return render_template('upload.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def predict(image):
    # 이미지 전처리
    transform = transforms.Compose([
        # 흑백 이미지로 변환
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))  # 이미지 평탄화
    ])
    image = transform(image).unsqueeze(0).to(model.device)
    # 예측
    output = model.model(image)
    _, predicted = torch.max(output.data, 1)
    return str(predicted.item())

if __name__ == '__main__':
    app.run(debug=True)
```

<br>

## 결론

PyTorch를 이용해 신경망을 구성하고 이를 실행하는 예시에 대해 알았습니다. 이후 모델, 모델을 실행하는 부분, 사용자 부분을 변형 해 볼 수 있습니다.

<br>

아래는 인공지능을 공부하고 개발하기 위한 참고사항입니다:

* [PyTorch 튜토리얼](https://tutorials.pytorch.kr/?\_gl=1\*16pumw7\*\_ga\*MTg4Mzg4MDMxMy4xNzEzMjYxMjY0\*\_ga_LEHG248408\*MTcxMzI2MjU4MC4xLjEuMTcxMzI2MjU4NC41Ni4wLjA.\*\_ga_L5NC8SBFPY\*MTcxMzI2MjU4MC4xLjEuMTcxMzI2MjU4NC41Ni4wLjA.\*\_ga_LZRD6GXDLF\*MTcxMzI2MTI2My4xLjEuMTcxMzI2MjU4NC41NC4wLjA.)에서 보다 많은 튜토리얼을 볼 수 있고 GPT의 도움을 받아 만들어 볼 수 있습니다.
* 딥러닝 모델(Model)는 CNNs, RNNs, GANs등이 있습니다. 예를 들어 GAN에는 생성자(Generator)와 판별자(Discriminator) 모듈이 있습니다.
* 레이어(Layer)는 Dense Layer, Convolutional Layer, Pooling Layer, Recurrent Layer, Normalization Layer, Dropout Layer등이 있습니다.
* 손실 함수(Loss Function)는 Mean Squared Error, Cross-Entropy Loss, Hinge Loss, Humber Loss, Log-Cosh Loss등이 있습니다.
* 옵티마이저(Optimizer)는 Adagrad, Adadelta, RMSProp, Momentum, Nesterov Accelerated Gradient, Adam등이 있습니다.
* 인공지능의 트랜드를 알기 위해선 [HuggingFace](https://huggingface.co/)의 트랜드를 참고할 수 있고, 특정 데이터 셋을 이용하여 인공지능을 구현하는 방법 등이 필요한 경우 [Kaggle](https://www.kaggle.com/)에서 찾아보고 Kaggle의 Notebook을 이용하여 실행하여 공부할 수 있습니다.

<br>

### 참고 자료

* 서지영, 『딥러닝 파이토치』 교과서, (주)도서출판길벗, 2022년 3월
* [파이토치 한글 사용자 모임, "파이토치(PYTORCH) 기본 익히기"](https://tutorials.pytorch.kr/beginner/basics/intro.html)
