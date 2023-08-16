import pandas as pd
import scipy.io


def read_files(csv_path, mat_path):
    # Read CSV file
    df = pd.read_csv(csv_path)

    # Read MATLAB file
    mat_data = scipy.io.loadmat(mat_path)

    return df, mat_data

csv_path = '/media/mmohseni/ubuntu/projects/bp-benchmark/datasets/org_to_splitted/uci2_dataset/feat_fold_0.csv'
mat_path = '/media/mmohseni/ubuntu/projects/bp-benchmark/datasets/org_to_splitted/uci2_dataset/signal_fold_0.mat'



df, mat_data = read_files(csv_path, mat_path)


print('Done')