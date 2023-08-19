import scipy.io
import numpy as np


def concatenate_mat_files():
    # Define the paths to the .mat files
    mat_path_1 = '/media/mmohseni/ubuntu/Datasets/BP_AllData/UCI_ourPreprocessed/signal_fold_0.mat'
    mat_path_2 = '/media/mmohseni/ubuntu/Datasets/BP_AllData/UCI_ourPreprocessed/signal_fold_1.mat'
    mat_path_3 = '/media/mmohseni/ubuntu/Datasets/BP_AllData/UCI_ourPreprocessed/signal_fold_2.mat'

    # Load data from each .mat file
    data_1 = scipy.io.loadmat(mat_path_1)
    data_2 = scipy.io.loadmat(mat_path_2)
    data_3 = scipy.io.loadmat(mat_path_3)

    # Extract the necessary data from each dictionary
    abp_per_tot_1 = data_1['ABP_Per_tot']
    dp_1 = data_1['DP']
    sp_1 = data_1['SP']
    abp_signal_1 = data_1['abp_signal']
    signal_1 = data_1['signal']

    abp_per_tot_2 = data_2['ABP_Per_tot']
    dp_2 = data_2['DP']
    sp_2 = data_2['SP']
    abp_signal_2 = data_2['abp_signal']
    signal_2 = data_2['signal']

    abp_per_tot_3 = data_3['ABP_Per_tot']
    dp_3 = data_3['DP']
    sp_3 = data_3['SP']
    abp_signal_3 = data_3['abp_signal']
    signal_3 = data_3['signal']

    # Concatenate the data from all files
    # combined_abp_per_tot = np.concatenate((abp_per_tot_1, abp_per_tot_2, abp_per_tot_3), axis=0)
    combined_dp = np.concatenate((dp_1, dp_2, dp_3), axis=0)
    combined_sp = np.concatenate((sp_1, sp_2, sp_3), axis=0)
    combined_abp_signal = np.concatenate((abp_signal_1, abp_signal_2, abp_signal_3), axis=0)
    combined_signal = np.concatenate((signal_1, signal_2, signal_3), axis=0)

    # Create a new dictionary to hold the combined data
    combined_data = {
        # 'ABP_Per_tot': combined_abp_per_tot,
        'DP': combined_dp,
        'SP': combined_sp,
        'abp_signal': combined_abp_signal,
        'signal': combined_signal
    }

    # Save the combined data as a new .mat file
    combined_mat_path = '/media/mmohseni/ubuntu/Datasets/BP_AllData/UCI_ourPreprocessed/combined_data.mat'
    scipy.io.savemat(combined_mat_path, combined_data)


def read_mat_file():
    mat_path = 'valid_data.mat'


    data = scipy.io.loadmat(mat_path)

    print(len(data['signal']))

def splits_train_val_test():
    import scipy.io
    import numpy as np

    # Load the .mat file
    mat_path_1 = '/media/mmohseni/ubuntu/Datasets/BP_AllData/UCI_ourPreprocessed/combined_data.mat'
    data = scipy.io.loadmat(mat_path_1)

    # Extract the data
    signal = data['signal']
    abp_signal = data['abp_signal']
    SP = data['SP']
    DP = data['DP']

    # Calculate the indices for splitting
    train_idx = int(0.7 * signal.shape[0])
    valid_idx = train_idx + int(0.2 * signal.shape[0])

    # Randomly shuffle the indices
    indices = np.random.permutation(signal.shape[0])

    # Split the data into train, valid and test datasets
    train_data = {k: data[k][indices[:train_idx]] for k in ['signal', 'abp_signal', 'SP', 'DP']}
    valid_data = {k: data[k][indices[train_idx:valid_idx]] for k in ['signal', 'abp_signal', 'SP', 'DP']}
    test_data = {k: data[k][indices[valid_idx:]] for k in ['signal', 'abp_signal', 'SP', 'DP']}

    # Save the new .mat files
    scipy.io.savemat('train_data.mat', train_data)
    scipy.io.savemat('valid_data.mat', valid_data)
    scipy.io.savemat('test_data.mat', test_data)





if __name__ == "__main__":
    # concatenate_mat_files()
    read_mat_file()
    # splits_train_val_test()
