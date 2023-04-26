import numpy as np


class ID3:
    def calc_total_entropy(self, X, label, class_list):
        n_rows = X.shape[0]  # the total size of the dataset
        cl_entropy = []
        print('Подсчитаем общую энтропию разбиения: ')
        for cl in class_list:  # for each class in the label
            # number of the class
            cl_count = X[X[label] == cl].shape[0]
            p_cl = cl_count/n_rows
            # entropy of the class
            cl_entropy.append(- (p_cl) * np.log2(p_cl))
            # adding the class entropy to the total entropy of the dataset
            print(
                f'Info({cl}) = {cl_count} - ({cl_count} / {n_rows}) * log₂({cl_count} / {n_rows}) = {cl_entropy[-1]}')
        total_entropy = sum(cl_entropy)
        print('Info(T) =', ' + '.join(np.around(cl_entropy, 5).astype(str)),
              ' = ', round(total_entropy, 5))
        return total_entropy

    def calc_entropy(self, feature_value_data, label, class_list, feature_name):
        cl_count = feature_value_data.shape[0]
        entropy = 0
        entropy_class = []
        for cl in class_list:
            # row count of class c
            label_cl_count = feature_value_data[feature_value_data[label]
                                                == cl].shape[0]
            if label_cl_count != 0:
                p_cl = label_cl_count/cl_count  # probability of the class
                entropy_class.append(- p_cl * np.log2(p_cl))
                print(f'Infoₛ({cl}|{feature_name})',
                      f'- {round(p_cl, 5)} * log₂({label_cl_count} / {cl_count})',
                      f'{round(entropy_class[-1], 5)}', sep=' = ')
        entropy = sum(entropy_class)
        print(f'Infoₛ({feature_name})',
              ' + '.join(np.around(entropy_class, 5).astype(str)),
              entropy, sep=' = ')
        return entropy

    def calc_info_gain(self, feature_name, X, label, class_list):
        # unqiue values of the feature
        feature_value_list = X[feature_name].unique()
        n_rows = X.shape[0]
        feature_info = 0.0

        print(f'Подсчитаем прирост информации для {feature_name}')

        for feature_value in feature_value_list:
            # filtering rows with that feature_value
            feature_value_data = X[X[feature_name]
                                   == feature_value]
            feature_value_count = feature_value_data.shape[0]
            # calculcating entropy for the feature value
            print('Подсчитаем энтропию:')
            feature_value_entropy = self.calc_entropy(
                feature_value_data, label, class_list, feature_name)
            feature_value_probability = feature_value_count/n_rows
            print(
                f'P({feature_value}|{feature_name}) = {feature_value_count} / {n_rows} = {round(feature_value_probability, 5)}')
            # calculating information of the feature value
            feature_info += feature_value_probability * feature_value_entropy
        print('Infoₛ(T)',
              ' + '.join([f'P({v}|{feature_name}) * Infoₛ({feature_name})'
                          for v in feature_value_list]),
              round(feature_info, 5), sep=' = ')
        total_entropy = self.calc_total_entropy(X, label, class_list)
        # calculating information gain by subtracting
        gain = total_entropy - feature_info
        print('Gain(S)', 'Info(T) - Infoₛ(T)',
              f'{total_entropy} - {feature_info}', gain, sep=' = ')
        return gain

    def find_most_informative_feature(self, X, label, class_list):
        # finding the feature names in the dataset
        feature_list = X.columns.drop(label)
        # N.B. label is not a feature, so dropping it
        max_info_gain = -1
        max_info_feature = None

        for feature in feature_list:  # for each feature in the dataset
            feature_info_gain = self.calc_info_gain(
                feature, X, label, class_list)
            if max_info_gain < feature_info_gain:  # selecting feature name with highest information gain
                max_info_gain = feature_info_gain
                max_info_feature = feature
        print('Максимальный Gain найден у', max_info_feature)

        return max_info_feature

    def generate_sub_tree(self, feature_name, X, label, class_list):
        feature_value_count_dict = X[feature_name].value_counts(
            sort=False).to_dict()  # dictionary of the count of unqiue feature value
        tree = {}  # sub tree or node

        for feature_value, count in feature_value_count_dict.items():
            # dataset with only feature_name = feature_value
            feature_value_data = X[X[feature_name]
                                   == feature_value]

            assigned_to_node = False  # flag for tracking feature_value is pure class or not
            for cl in class_list:  # for each class
                # count of class c
                class_count = feature_value_data[feature_value_data[label]
                                                 == cl].shape[0]

                # count of (feature_value = count) of class (pure class)
                if class_count == count:
                    # print(f'Все записи при {feature_name}={feature_value}',
                    #       f'принадлежат классу {cl}',
                    #       f'=> добавляем в поддерево {feature_name}:')
                    tree[feature_value] = cl  # adding node to the tree
                    # removing rows with feature_value
                    X = X[X[feature_name]
                          != feature_value]
                    assigned_to_node = True
                    # print(tree)
            if not assigned_to_node:  # not pure class
                # as feature_value is not a pure class, it should be expanded further,
                tree[feature_value] = "?"
                # so the branch is marking with ?
            # print(f'Поддерево {feature_name} будет выглядеть так:')
            # print(tree)

        return tree, X

    def make_tree(self, root, prev_feature_value, X, label, class_list):
        if X.shape[0] != 0:  # if dataset becomes enpty after updating
            max_info_feature = self.find_most_informative_feature(
                X, label, class_list)  # most informative feature
            tree, X = self.generate_sub_tree(
                max_info_feature, X, label, class_list)  # getting tree node and updated dataset
            next_root = None

            if prev_feature_value != None:  # add to intermediate node of the tree
                root[prev_feature_value] = dict()
                root[prev_feature_value][max_info_feature] = tree
                next_root = root[prev_feature_value][max_info_feature]
            else:  # add to root of the tree
                root[max_info_feature] = tree
                next_root = root[max_info_feature]

            for node, branch in list(next_root.items()):  # iterating the tree node
                if branch == "?":  # if it is expandable
                    # using the updated dataset
                    feature_value_data = X[X[max_info_feature] == node]
                    # recursive call with updated dataset

                    self.make_tree(next_root, node,
                                   feature_value_data, label, class_list)

    def fit(self, X, label):
        self.tree = {}  # tree which will be updated
        # getting unqiue classes of the label
        X_train = X.copy()
        class_list = X_train[label].unique()
        self.make_tree(self.tree, None, X_train, label, class_list)
        print('Итак, дерево приняло вид:')
        print(self.tree)
        return self.tree

    def predict(self, X, tree=None):
        if tree is None:
            tree = self.tree
        if type(tree) != dict:
            return tree
        root_node = next(iter(tree))
        feature_value = X[root_node]
        if feature_value in tree[root_node]:
            return self.predict(X, tree[root_node][feature_value])
        else:
            return None

    def test(self, X, label_col):
        preds = np.array([self.predict(X.iloc[i], self.tree) for i in X.index])
        acc = (preds == X[label_col].values).mean()
        return preds, acc
