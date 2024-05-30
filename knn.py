from DataEmbedding import DataEmbedding
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


train_path = "train.tagged"
test_path = "test.untagged"
dev_path = "dev.tagged"

def main():
    f1_list = []
    for k in range(10):
        embedding = DataEmbedding("glove", 200)
        X_train, y_train = embedding.get_embedding_from_file(train_path, True, 'knn')
        classifer = KNeighborsClassifier(n_neighbors=k+1, metric='cosine')
        classifer.fit(X_train, y_train)
        X_test, y_test = embedding.get_embedding_from_file(dev_path, True, 'knn')
        pred = classifer.predict(X_test)
        f1 = f1_score(y_test, pred, pos_label=1)
        print(f1)
        f1_list.append(f1)
    print(f1_list)


if __name__ == "__main__":
    main()


