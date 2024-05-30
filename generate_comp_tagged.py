from gensim import downloader
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.utils.rnn as rnn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import f1_score


train_path = "train.tagged"
test_path = "test.untagged"
dev_path = "dev.tagged"


class DataEmbedding:
    def __init__(self, embedding_type, vec_size):

        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


        self.embedding_type = embedding_type
        self.vec_size = vec_size
        self.embedding_path = "glove-twitter-200"
        self.embedding_model = downloader.load(self.embedding_path)
        self.unknown_word = torch.rand(self.vec_size, requires_grad=True)

    def get_embedding_from_file(self, file_path, is_tagged, type):
        tags = []
        words = []
        with open(file_path, encoding="utf-8") as file:
            for line in file:
                if len(line.split("\t")) == 2:
                    if is_tagged:
                        tag = line.split("\t")[1]
                        if tag[0] == 'O':
                            tag = 0
                        else:
                            tag = 1
                        tags.append(tag)
                    words.append(line.split("\t")[0].lower())
        X = []
        if type == 'knn':
            for word in words:
                if word in self.embedding_model.vocab:
                    X.append((self.embedding_model[word]))
                else:
                    X.append(self.unknown_word.detach().numpy())
            return np.array(X), tags

        if type == 'nn':
            for word in words:
                if word in self.embedding_model.vocab:
                    X.append(torch.tensor(self.embedding_model[word]))
                else:
                    X.append(self.unknown_word)
            return torch.stack(X), tags

    def get_dataloader_from_embedding(self, X, y, batch_size):
        y = torch.tensor(y)
        data_set = TensorDataset(X, y)
        data_loader = DataLoader(data_set, batch_size=batch_size)
        return data_loader

    def get_sen_embedding_from_path(self, file_path, is_tagged):
        X = []
        y = []
        with open(file_path, encoding="utf-8") as file:
            sentence_vec = []
            sen_tags = []
            sen_lens = []
            for line in file:
                if line != "\t\n" and line != "" and line != "\n" and line != "\ufeff\n":
                    word = line.split("\t")[0].lower()
                    if is_tagged:
                        tag = 1 if line.split("\t")[1][0] != 'O' else 0
                        sen_tags.append(tag)
                    if word in self.embedding_model.vocab:
                        sentence_vec.append(torch.tensor(self.embedding_model[word]))
                    else:
                        sentence_vec.append(self.unknown_word)
                else:
                    if "train" in file_path:
                        if 1 in sen_tags:
                            sentence_vec = torch.stack(sentence_vec)
                            X.append(sentence_vec)
                            y.append(sen_tags)
                            sen_lens.append(len(sentence_vec))
                    elif "train" not in file_path:
                        sentence_vec = torch.stack(sentence_vec)
                        X.append(sentence_vec)
                        y.append(sen_tags)
                        sen_lens.append(len(sentence_vec))
                    sentence_vec = []
                    sen_tags = []

        X = rnn.pad_sequence(X, batch_first=True, padding_value=0)
        if is_tagged:
            list_tags = y
            list_tags = [x for xs in list_tags for x in xs]
            y = [torch.Tensor([0 if y_ == 0 else 1 for y_ in sentence_tags])for sentence_tags in y]
            y = rnn.pad_sequence(y, batch_first=True, padding_value=0)
        data_set = [*zip(X, y, sen_lens)]
        torch.manual_seed(42)
        data_loader = DataLoader(data_set, batch_size=32)
        if is_tagged:
            return data_loader, list_tags
        return data_loader

    def get_sen_embedding_from_path_test(self, file_path):
        X = []
        with open(file_path, encoding="utf-8") as file:
            sentence_vec = []
            sen_lens = []
            for line in file:
                line = line.strip()
                if line != "\t\n" and line != "" and line != "\n" and line != "\ufeff\n":
                    word = line.split("\t")[0].lower()
                    print(word)
                    if word in self.embedding_model.vocab:
                        sentence_vec.append(torch.tensor(self.embedding_model[word]))
                    else:
                        sentence_vec.append(self.unknown_word)
                else:
                    sentence_vec = torch.stack(sentence_vec)
                    X.append(sentence_vec)
                    sen_lens.append(len(sentence_vec))
                    sentence_vec = []
                    print("next sen")

        X = rnn.pad_sequence(X, batch_first=True, padding_value=0)
        data_set = [*zip(X, sen_lens)]
        torch.manual_seed(42)
        data_loader = DataLoader(data_set, batch_size=32)
        return data_loader

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
    lr = 0.001
    print("Creating model...")
    batch_size = 32
    dropout = 0.2
    num_layers = 2
    num_of_classes = 2
    hidden_size = 128
    weight = torch.tensor([0.2, 0.8]).to(device)
    loss_function = nn.CrossEntropyLoss(weight=weight)
    embedding = DataEmbedding("glove", vec_size)
    lstm_model = LSTM(vec_size, num_of_classes, hidden_size, dropout, num_layers)
    lstm_model.to(device)
    optimizer = Adam(params=lstm_model.parameters(), lr=lr)
    train_loader, _ = embedding.get_sen_embedding_from_path(train_path, True)
    dev_loader, y = embedding.get_sen_embedding_from_path(dev_path, True)
    test_loader = embedding.get_sen_embedding_from_path_test(test_path)

    max_f1 = 0
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
            loss = loss_function(o, labels_one_hot.float())
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()


        all_preds = []
        for sens, labels, sen_lens in dev_loader:
            sens = sens.to(device)
            oo = lstm_model(sens, sen_lens)
            preds = np.array(oo.detach().cpu().numpy())
            all_preds.append(preds)

        all_preds = np.concatenate(all_preds, axis=0)
        all_preds = [np.argmax(pred) for pred in all_preds]
        f1 = f1_score(y, all_preds, pos_label=1)
        if f1 > max_f1:
            max_f1 = f1
            torch.save(lstm_model.state_dict(), 'model_weights.pth')

    lstm_model.load_state_dict(torch.load('model_weights.pth')) #loading best weights
    print("best f1 score on dev ", max_f1)
    all_preds = []
    print("predcting and tagging test.untagged")
    for sens, sen_lens in test_loader:
        sens = sens.to(device)
        oo = lstm_model(sens, sen_lens)
        preds = np.array(oo.detach().cpu().numpy())
        all_preds.append(preds)

    all_preds = np.concatenate(all_preds, axis=0)
    all_preds = [np.argmax(pred) for pred in all_preds]

    idx = 0
    with open(test_path, encoding="utf-8") as file:
        lines = file.readlines()

    with open('googlenews.txt', encoding="utf-8", mode='w') as output_file:
        for line in lines:
            line = line

            if line != "\t\n" and line != "" and line != "\n" and line != "\ufeff\n":  # Check if the line is not blank
                line = line.strip()
                word = line
                tag = all_preds[idx]
                output_file.write(f'{word}\t{tag}')
                output_file.write('\n')# Write the word and its tag to the output file
                idx += 1
            else:  # If the line is blank, write it as is to the output file
                output_file.write('\n')








if __name__ == "__main__":
    main()