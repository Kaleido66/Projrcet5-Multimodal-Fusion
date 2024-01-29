from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

from transformers import AdamW, get_linear_schedule_with_warmup, AutoModel, AutoTokenizer

with open('./train.txt', 'r') as f:
    lines = f.readlines()

train_set = []

for line in lines[1:]:
    data = {}
    line = line.replace('\n', '')
    guid, tag = line.split(',')
    if tag == 'positive':
        label = 0
    elif tag == 'neutral':
        label = 1
    else:
        label = 2
    data['guid'] = guid
    data['label'] = label
    train_set.append(data)

# print(len(train_set)) # 4000
# print(train_set)

with open('./test_without_label.txt', 'r') as f:
    lines = f.readlines()

test_set = []
for line in lines[1:]:
    data = {}
    data['guid'] = line.split(',')[0]
    test_set.append(data)


# def data_process(dataset):
#     for data in dataset:
#         guid = data['guid']
#         image_path = './data/' + guid + '.jpg'
#         image = Image.open(image_path).convert('RGB')
#         array = np.array(image.resize((224, 224)))
#         data['image'] = array.reshape((3, 224, 224))
#
#         text_path = './data/' + guid + '.txt'
#         f = open(text_path, 'r', errors='ignore')
#         lines = f.readlines()
#         # print(lines)
#         text = ''
#         for line in lines:
#             text += line
#         data['text'] = text

# 修改 data_process 函数以将 numpy ndarray 列表转换为单个 numpy ndarray
def data_process(dataset):
    for data in dataset:
        guid = data['guid']
        image_path = './data/' + guid + '.jpg'
        image = Image.open(image_path).convert('RGB')
        array = np.array(image.resize((224, 224)))
        data['image'] = array.reshape((3, 224, 224))

        text_path = './data/' + guid + '.txt'
        with open(text_path, 'r', errors='ignore') as f:
            lines = f.readlines()
            text = ''.join(lines)
        data['text'] = text

data_process(train_set)
data_process(test_set)

train_set_num = 3500
valid_set_num = 500
train_set, valid_set = random_split(train_set, [train_set_num, valid_set_num])

data_process(valid_set)

class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=(3, 3), padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=(3, 3), padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = F.relu(output)
        output = self.conv2(x)
        output = self.bn2(output)
        output = F.relu(output + x)
        return output


class ShortcutResBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(ShortcutResBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=(1, 1), stride=2)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(input_channel, output_channel, kernel_size=(3, 3), padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, kernel_size=(3, 3), padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        output1 = self.conv1(x)
        output1 = self.bn1(output1)
        output2 = self.conv2(x)
        output2 = self.bn2(output2)
        output2 = F.relu(output2)
        output2 = self.conv3(output2)
        output2 = self.bn3(output2)
        output = F.relu(output1 + output2)
        return output


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), padding=3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), padding=1, stride=2)
        self.res1 = ResBlock(64, 64)
        self.res2 = ResBlock(64, 64)
        self.shortcut1 = ShortcutResBlock(64, 128)
        self.res3 = ResBlock(128, 128)
        self.shortcut2 = ShortcutResBlock(128, 256)
        self.res4 = ResBlock(256, 256)
        self.shortcut3 = ShortcutResBlock(256, 512)
        self.res5 = ResBlock(512, 512)
        self.pool2 = nn.AvgPool2d((7, 7))
        self.dropout = nn.Dropout(0)
        self.fc = nn.Linear(512, 3)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.pool1(F.relu(output))
        output = self.res1(output)
        output = self.res2(output)
        output = self.shortcut1(output)
        output = self.res3(output)
        output = self.shortcut2(output)
        output = self.res4(output)
        output = self.shortcut3(output)
        output = self.res5(output)
        output = self.pool2(output)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        return output


image_train = []
image_train_labels = []
image_valid = []
image_valid_labels = []

for data in train_set:
    image_train.append(data['image'])
    image_train_labels.append(data['label'])

for data in valid_set:
    image_valid.append(data['image'])
    image_valid_labels.append(data['label'])

image_train = torch.from_numpy(np.array(image_train))
image_train_labels = torch.from_numpy(np.array(image_train_labels))
image_valid = torch.from_numpy(np.array(image_valid))
image_valid_labels = torch.from_numpy(np.array(image_valid_labels))

train_loader = DataLoader(TensorDataset(image_train, image_train_labels), batch_size=100, shuffle=True)
valid_loader = DataLoader(TensorDataset(image_valid, image_valid_labels), batch_size=50)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_classifier = ResNet18()
image_classifier.to(device)

epoch_num = 50
learning_rate = 1e-6
total_step = epoch_num * len(train_loader)

optimizer = AdamW(image_classifier.parameters(), lr=learning_rate, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_step, num_training_steps=total_step)

# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(epoch_num):
  running_loss = 0
  for i, data in enumerate(train_loader):
    inputs, labels = data
    inputs = inputs.float()
    inputs = inputs.to(device)
    labels = labels.to(device)
    # print(inputs.shape)
    outputs = image_classifier(inputs)
    # print(outputs.shape)

    labels = torch.tensor(labels, dtype=torch.int64)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    running_loss += loss.item()
  print('epoch: %d  loss: %.3f' % (epoch+1, running_loss / 35))
  running_loss = 0

correct_num = 0
total_num = 0
with torch.no_grad():
  for data in valid_loader:
      inputs, answers = data
      inputs = inputs.float()
      inputs = inputs.to(device)
      answers = answers.to(device)
      outputs = image_classifier(inputs)
      _, predicted = torch.max(outputs.data, 1)
      for i in range(len(predicted.tolist())):
          total_num += answers.size(0)
          correct_num += (predicted == answers).sum().item()

print('Training Accuracy: %.3f%%' % (100 * correct_num / total_num))


# checkpoint = 'bert-base-chinese'
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# bert_model = AutoModel.from_pretrained(checkpoint)

checkpoint = './bert-base-chinese'
# 使用离线模式加载预训练的BERT模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_auth_token=False)
bert_model = AutoModel.from_pretrained(checkpoint, use_auth_token=False, ignore_mismatched_sizes=True)
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# bert_model = AutoModel.from_pretrained(checkpoint)
# bert_model.to(device)

class TextClassifier(nn.Module):
    def __init__(self):
        super(TextClassifier, self).__init__()
        self.model = bert_model
        self.model = self.model.to(device)
        self.dropout = nn.Dropout(0)
        # self.model.to(device)
        self.fc = nn.Linear(768, 3)

    def forward(self, x, attn_mask=None):
        x = x.to(device)
        attn_mask = attn_mask.to(device)
        output = self.model(x, attention_mask=attn_mask)
        # output = output.to(device)
        output = output[1]
        output = torch.flatten(output, 1)
        output = self.fc(output)
        return output

text_train = []
text_valid = []

for data in train_set:
    tokenized_text = tokenizer(data['text'], max_length=128, padding='max_length', truncation=True)
    # tokenized_text['input_ids'] = torch.from_numpy(np.array(tokenized_text['input_ids']))
    tokenized_text['label'] = data['label']
    text_train.append(tokenized_text)

for data in valid_set:
    tokenized_text = tokenizer(data['text'], max_length=128, padding='max_length', truncation=True)
    tokenized_text['label'] = data['label']
    text_valid.append(tokenized_text)


class TextDataset(Dataset):
    def __init__(self, data):
        super(TextDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = self.data[idx]['input_ids']
        attn_mask = self.data[idx]['attention_mask']
        label = self.data[idx]['label']
        return input_ids, attn_mask, label


train_loader = DataLoader(TextDataset(text_train), batch_size=25, shuffle=True)
valid_loader = DataLoader(TextDataset(text_valid), batch_size=25)

text_classifier = TextClassifier()
text_classifier.to(device)
# classifier.model.to(device)

epoch_num = 20
learning_rate = 1e-5
total_step = epoch_num * len(train_loader)

optimizer = AdamW(text_classifier.parameters(), lr=learning_rate, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_step, num_training_steps=total_step)

criterion = nn.CrossEntropyLoss()

# classifier.train()

for epoch in range(epoch_num):
    running_loss = 0
    for i, data in enumerate(train_loader):
        input_ids, attn_mask, labels = data

        input_ids = torch.tensor([item.numpy() for item in input_ids])
        # input_ids = torch.tensor(np.array(input_ids))
        attn_mask = torch.tensor([item.numpy() for item in attn_mask])
        input_ids = input_ids.T
        attn_mask = attn_mask.T
        # labels = torch.tensor([item.numpy() for item in labels])
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        # print(input_ids.shape)
        # print(attn_mask.shape)

        outputs = text_classifier(input_ids, attn_mask)
        # print(outputs.shape)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
    print('epoch: %d  loss: %.3f' % (epoch + 1, running_loss / 140))
    running_loss = 0

orrect_num = 0
total_num = 0
with torch.no_grad():
    for data in valid_loader:
        input_ids, attn_mask, labels = data
        input_ids = torch.tensor([item.numpy() for item in input_ids])
        input_ids = input_ids.T
        attn_mask = torch.tensor([item.numpy() for item in attn_mask])
        attn_mask = attn_mask.T
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        outputs = text_classifier(input_ids, attn_mask)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(len(predicted.tolist())):
            total_num += labels.size(0)
            correct_num += (predicted == labels).sum().item()

print('Training Accuracy: %.3f%%' % (100 * correct_num / total_num))


class MultimodalDataset(Dataset):
    def __init__(self, data):
        super(MultimodalDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        guid = self.data[idx]['guid']
        input_ids = torch.tensor(self.data[idx]['input_ids'])
        attn_mask = torch.tensor(self.data[idx]['attn_mask'])
        image = torch.tensor(self.data[idx]['image'])
        label = self.data[idx].get('label')
        if label is None:
            label = -100
        label = torch.tensor(label)
        return guid, input_ids, attn_mask, image, label


def dataset_process(dataset):
    for data in dataset:
        tokenized_text = tokenizer(data['text'], max_length=128, padding='max_length', truncation=True)
        data['input_ids'] = tokenized_text['input_ids']
        data['attn_mask'] = tokenized_text['attention_mask']


dataset_process(train_set)
dataset_process(valid_set)
dataset_process(test_set)

train_loader = DataLoader(MultimodalDataset(train_set), batch_size=25, shuffle=True)
valid_loader = DataLoader(MultimodalDataset(valid_set), batch_size=25)
test_loader = DataLoader(MultimodalDataset(test_set), batch_size=25)


class MultimodalModel(nn.Module):
    def __init__(self, image_classifier, text_classifier, output_features, image_weight=0.5, text_weight=0.5):
        super(MultimodalModel, self).__init__()
        self.image_classifier = image_classifier
        self.text_classifier = text_classifier
        # 将最后的全连接层删除
        self.image_classifier.fc = nn.Sequential()  # (batch_num, 512)
        self.text_classifier.fc = nn.Sequential()  # (batch_num, 768)
        # 文本特征向量和图片特征向量的权重, 默认均为0.5
        self.image_weight = image_weight
        self.text_weight = text_weight
        self.fc1 = nn.Linear((512 + 768), output_features)
        self.fc2 = nn.Linear(output_features, 3)  # 输出类别数为3，你的任务中可能需要调整这个值

    def forward(self, image, input_ids, attn_mask):
        image_output = self.image_classifier(image)
        text_output = self.text_classifier(input_ids, attn_mask)
        output = torch.cat([image_output, text_output], dim=-1)
        output = self.fc1(output)
        output = self.fc2(output)
        return output

multimodal_model = MultimodalModel(image_classifier=image_classifier, text_classifier=text_classifier,
                                   output_features=100, image_weight=0.5, text_weight=0.5)
multimodal_model.to(device)

epoch_num = 10
learning_rate = 1e-5
total_step = epoch_num * len(train_loader)

# 定义优化器和学习率调度器
optimizer = AdamW(multimodal_model.parameters(), lr=1e-5)
total_step = epoch_num * len(train_loader)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_step, num_training_steps=total_step)
# optimizer = AdamW(multimodal_model.parameters(), lr=learning_rate, eps=1e-8)
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_step, num_training_steps=total_step)
criterion = nn.CrossEntropyLoss()

for epoch in range(epoch_num):
  running_loss = 0
  for i, data in enumerate(train_loader):
    _, input_ids, attn_mask, image, label = data
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)
    image = image.to(device)
    image = image.float()
    label = label.to(device)

    outputs = multimodal_model(input_ids=input_ids, attn_mask=attn_mask, image=image)
    # print(outputs.shape)
    loss = criterion(outputs, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    running_loss += loss.item()
  print('epoch: %d  loss: %.3f' % (epoch+1, running_loss/140))
  running_loss = 0

correct_num = 0
total_num = 0
with torch.no_grad():
  for data in valid_loader:
      _, input_ids, attn_mask, image, label = data
      input_ids = input_ids.to(device)
      attn_mask = attn_mask.to(device)
      image = image.to(device)
      image = image.float()
      label = label.to(device)

      outputs = multimodal_model(input_ids=input_ids, attn_mask=attn_mask, image=image)
      _, predicted = torch.max(outputs.data, 1)
      for i in range(len(predicted.tolist())):
          total_num += label.size(0)
          correct_num += (predicted == label).sum().item()

print('Training Accuracy: %.3f%%' % (100 * correct_num / total_num))

# 保存模型
torch.save(multimodal_model.state_dict(), 'multimodal_model.pth')

test_dict = {}
with torch.no_grad():
    for data in test_loader:
        guid, input_ids, attn_mask, image, label = data
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        image = image.to(device)
        image = image.float()
        label = label.to(device)

        outputs = multimodal_model(input_ids=input_ids, attn_mask=attn_mask, image=image)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.tolist()
        for i in range(len(predicted)):
            id = guid[i]
            test_dict[id] = predicted[i]

with open('./test_with_label.txt', 'r') as f:
    lines = f.readlines()

f1 = open('./test.txt', 'w')
f1.write(lines[0])

for line in lines[1:]:
    # print(line)
    guid = line.split(',')[0]
    f1.write(guid)
    f1.write(',')
    label = test_dict[guid]
    if label == 0:
        f1.write('positive\n')
    elif label == 1:
        f1.write('neutral\n')
    else:
        f1.write('negative\n')
