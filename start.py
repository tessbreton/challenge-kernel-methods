import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tee import StdoutTee
from time import strftime
import os, yaml, warnings
warnings.filterwarnings("ignore")

from svm import SVM
from kernels import MismatchKernel, SpectrumKernel, SubstringKernel


KERNEL_CLASSES = {"spectrum": SpectrumKernel, "mismatch": MismatchKernel, "substring": SubstringKernel}
C_VALUES = {'0': 0.3, '1': 0.5, '2': 0.33}

def load_kernel(kernel, dataset, params, df, df_test):
    filename = f"{kernel}_{'_'.join(map(str, params))}"
    kernel_path_train, kernel_path_test = f'kernel_matrices/{filename}_train_{dataset}.npy', f'kernel_matrices/{filename}_test_{dataset}.npy'

    try:
        K_train, K_test = np.load(kernel_path_train), np.load(kernel_path_test)
        print(f"Successfully loaded {kernel} kernel matrix from {kernel_path_train} and {kernel_path_test}")
    except IOError:
        print(f"No {kernel} kernel matrix found for the given parameters. Computing {kernel} kernel matrix...")
        kernel_instance = KERNEL_CLASSES[kernel](*params)
        K_train, K_test = kernel_instance.fit(df.seq.values), kernel_instance.predict(df_test.seq.values)
        np.save(kernel_path_train, K_train, allow_pickle=False)
        np.save(kernel_path_test, K_test, allow_pickle=False)
        print(f"Successfully saved {kernel} kernel matrix in {kernel_path_train} and {kernel_path_test}")
    return K_train, K_test


def load_data(dataset, train=True):
    df = pd.read_csv(f"data/X{'tr' if train else 'te'}{dataset}.csv", index_col=0)
    y = pd.read_csv(f"data/Ytr{dataset}.csv", index_col=0)['Bound'].values.ravel() if train else None
    return df, y


def compute_crossval(param, K_train, y_train):
    clf, cv = SVM(lbda=1/(2*2000*param)), StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_scores, val_scores = [], []

    for train_idx, val_idx in cv.split(K_train, y_train):
        K_train_split, K_val_split = K_train[train_idx][:, train_idx], K_train[val_idx][:, train_idx]
        clf.fit(K_train_split, y_train[train_idx])
        train_scores.append(clf.score(K_train_split, y_train[train_idx]))
        val_scores.append(clf.score(K_val_split, y_train[val_idx]))

    return np.mean(val_scores)


def compute_kernels(dataset, df, df_test):
    K_train, K_test = np.zeros((2000, 2000)), np.zeros((1000, 2000))

    with open("kernel_configs.yaml", "r") as file:
        kernel_configs = yaml.safe_load(file)

    for kernel, param_sets in kernel_configs.items():
        for params in param_sets.get(dataset, []):
            K_train_part, K_test_part = load_kernel(kernel, dataset, params, df, df_test)
            K_train += K_train_part
            K_test += K_test_part

    return K_train, K_test


def main_dataset(dataset):
    print('\n---------------- DATASET', dataset, '------------------------------------------------')
    df_train, y_train = load_data(dataset)
    df_test, _ = load_data(dataset, train=False)
    K_train, K_test = compute_kernels(dataset, df_train, df_test)

    C = C_VALUES[dataset]
    crossval_score = compute_crossval(C, K_train, y_train)
    print(crossval_score)

    clf = SVM(lbda=1/(2*2000*C)).fit(K_train, y_train)
    predictions_df = pd.Series(data=clf.predict(K_test), name='Bound', index=df_test.index)

    return predictions_df, crossval_score


def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("kernel_matrices", exist_ok=True)
    model_path = f'logs/{strftime("model_%d_%m_%H_%M")}'
    
    with StdoutTee(f"{model_path}.log", 'w', 1):
        results = [main_dataset(dataset) for dataset in ['0','1','2']]
        predictions, cv_scores = zip(*results)
        df_pred = pd.concat(predictions, axis=0)
        
        print('\n---------------- SUMMARY ------------------------------------------------')
        for dataset, score in zip(['0','1','2'], cv_scores):
            print(f'Dataset {dataset} : {score}')
        print(f'Average: {np.mean(cv_scores)}')

        df_pred.to_csv(f'{model_path}_submission.csv', header=True)


if __name__ == "__main__":
    main()