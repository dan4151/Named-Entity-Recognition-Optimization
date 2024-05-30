from DataEmbedding import DataEmbedding
import torch.nn as nn
import torch
import torch.optim as optim
from sklearn.metrics import f1_score
import numpy as np
from torch.optim import Adam

train_path = "train.tagged"
test_path = "test.untagged"
dev_path = "dev.tagged"


class NN(nn.Module):
    def __init__(self, size_input, num_of_classes, hidden_size, dropout):
        super(NN, self).__init__()
        self.num_of_classes = 2
        self.layer1 = nn.Linear(size_input, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_of_classes)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return x

def main():
    vec_size = 200
    lr = 0.01
    dropout = 0.2

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print("Creating model...")
    embedding = DataEmbedding("glove", vec_size)
    X_train, y_train = embedding.get_embedding_from_file(train_path, True, 'nn')
    train_loader = embedding.get_dataloader_from_embedding(X_train, y_train, batch_size=32)
    ff_model = NN(vec_size, 2, 64, dropout)
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(params=ff_model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Training on GPU.")
    else:
        print("No GPU available, training on CPU.")
    ff_model.to(device)
    X_dev, y_dev = embedding.get_embedding_from_file(dev_path, True, 'nn')
    f1_list = []
    epochs = 20
    for epoch in range(epochs):
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            o = ff_model.forward(inputs)
            loss = loss_function(o, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        X_dev = X_dev.to(device)
        pred = ff_model.forward(X_dev)
        pred = pred.cpu().detach().numpy()
        pred = [np.argmax(t) for t in pred]
        f1_score_ = f1_score(y_dev, pred, pos_label=1)
        print(f1_score_)
        f1_list.append([epoch, f1_score_])
    print(f1_list)

if __name__ == "__main__":
    main()

