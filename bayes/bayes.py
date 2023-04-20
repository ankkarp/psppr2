import numpy as np


class NaiveBayes:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.classes = np.unique(y)
        self.n_classes = len(y)
        self.n_samples, self.n_features = X.shape
        self.X = X
        self.y = y

    def get_class_probability(self, x, cl):
        product = 1
        cl_samples = self.X[self.y == cl]
        cl_size = cl_samples.shape[0]
        # print(f'P(X|C({cl})) = (', end='')
        for i in range(self.n_features):
            # print(
            #     f'({len(cl_samples[cl_samples[:, i] == x[i]])} / {cl_size})', end='')
            product *= len(cl_samples[cl_samples[:, i]
                           == x[i]]) / cl_size
        # print(
        #     f'* ({cl_size} / {self.n_classes}) = {product * (cl_size / self.n_samples)}')
        return product * (cl_size / self.n_samples)

    def predict(self, x):
        probs = np.array([self.get_class_probability(x, cl)
                         for cl in self.classes])
        order = np.argsort(probs)[::-1]
        # print(order, self.classes)
        return self.classes[order], probs[order]
