import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
from loss import F1_Loss
from DataEmbedding import DataEmbedding
import numpy as np
from sklearn.metrics import f1_score


train_path = "train.tagged"
test_path = "test.untagged"
dev_path = "dev.tagged"


class LSTM(nn.Module):
    def __init__(self, input_size, num_of_classes, hidden_size, dropout, layers):
        super().__init__()
        self.num_of_classes = num_of_classes
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.hidden_states = nn.Sequential(self.activation, nn.Linear(self.hidden_size * 2, self.hidden_size))
        self.hidden_states2 = nn.Sequential(nn.Linear(self.hidden_size, self.num_of_classes))

    def forward(self, sen_embeddings, sen_lens):
        packed = pack_padded_sequence(sen_embeddings, sen_lens, batch_first=True, enforce_sorted=False)
        lstm_packed_output, _ = self.lstm(input=packed)
        lstm_out, len_out = pad_packed_sequence(lstm_packed_output, batch_first=True)
        unpadded = []
        for x, len_x in zip(lstm_out, len_out):
            x_unpad = torch.Tensor(x[:len_x])
            unpadded.append(x_unpad)
        sentence = torch.concat(unpadded)
        x = self.hidden_states(sentence)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.hidden_states2(x)
        #x = nn.functional.sigmoid(x)
        #x = nn.functional.softmax(x, dim=1)
        return x


def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Training on GPU.")
    else:
        print("No GPU available, training on CPU.")

    vec_size = 200
    lr = 0.01
    print("Creating model...")
    batch_size = 32
    dropout = 0.2
    num_layers = 2
    num_of_classes = 2
    hidden_size = 254
    weight = torch.tensor([0.1, 0.9]).to(device)
    #loss_function = nn.CrossEntropyLoss(weight=weight)
    f1_loss = F1_Loss().cuda()
    embedding = DataEmbedding("glove", vec_size)
    lstm_model = LSTM(vec_size, num_of_classes, hidden_size, dropout, num_layers)
    lstm_model.to(device)
    optimizer = Adam(params=lstm_model.parameters(), lr=lr)
    train_loader, _ = embedding.get_sen_embedding_from_path(train_path, True)
    test_loader, y = embedding.get_sen_embedding_from_path(dev_path, True)

    epochs = 10
    for epoch in range(epochs):
        for index, (sens, labels, sen_lens) in enumerate(train_loader):
            sens = sens.to(device)
            labels = labels.to(device)
            o = lstm_model(sens, sen_lens)
            packed_labels = pack_padded_sequence(labels, sen_lens, batch_first=True, enforce_sorted=False)
            unpacked_labels, label_lens = pad_packed_sequence(packed_labels, batch_first=True)
            unpadded_labels = []
            for x, len_x in zip(unpacked_labels, label_lens):
                x_unpad = torch.Tensor(x[:len_x])
                unpadded_labels.append(x_unpad)
            unpadded_labels = torch.concat(unpadded_labels).long()
            labels_one_hot = nn.functional.one_hot(unpadded_labels, num_classes=num_of_classes)
            loss = f1_loss(o, labels_one_hot.float())
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()


        all_preds = []
        for sens, labels, sen_lens in test_loader:
            sens = sens.to(device)
            oo = lstm_model(sens, sen_lens)
            preds = np.array(oo.detach().cpu().numpy())
            all_preds.append(preds)

        all_preds = np.concatenate(all_preds, axis=0)
        all_preds = [np.argmax(pred) for pred in all_preds]
        print("epoch:", epoch, "f1_score=", f1_score(y, all_preds, pos_label=1))




if __name__ == "__main__":
    main()