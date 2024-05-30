from gensim import downloader
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.utils.rnn as rnn


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
                if word in self.embedding_model.key_to_index:
                    X.append((self.embedding_model[word]))
                else:
                    X.append(self.unknown_word.detach().numpy())
            return np.array(X), tags

        if type == 'nn':
            for word in words:
                if word in self.embedding_model.key_to_index:
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
                    if word in self.embedding_model.key_to_index:
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


