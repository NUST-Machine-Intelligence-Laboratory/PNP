import numpy as np
import matplotlib.pyplot as plt


def plot_results(result_file, layout='1x3'):
    if layout == '1x3':
        plot_results_1x3(result_file)
    elif layout == '2x2':
        plot_results_2x2(result_file)
    else:
        raise AssertionError(f'Unsupported layout `{layout}`')


def plot_results_1x3(result_file):
    fig, ax = plt.subplots(1, 3, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    metrics = ['Train Loss', 'Train Accuracy', 'Test Accuracy']
    with open(result_file, 'r') as f:
        lines = f.readlines()
    epoch_list = []
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    best_epoch, best_acc = 0, 0.0
    for line in lines:
        if not line.startswith('epoch'):
            continue
        line = line.strip()
        epoch, train_loss, train_acc, test_acc = line.split(' | ')[:4]
        epoch_list.append(int(epoch.split(': ')[1]))
        train_loss_list.append(float(train_loss.split(': ')[1]))
        train_acc_list.append(float(train_acc.split(': ')[1]))
        test_acc_list.append(float(test_acc.split(': ')[1]))
        best_accuracy_epoch = line.split(' | ')[-1]
        best_acc = float(best_accuracy_epoch.split(' @ ')[0].split(': ')[1])
        best_epoch = int(best_accuracy_epoch.split(' @ ')[1].split(': ')[1])
    results = [
        np.array(train_loss_list),
        np.array(train_acc_list),
        np.array(test_acc_list)
    ]

    for i in range(len(metrics)):
        ax[i].plot(epoch_list, results[i], '-', label=metrics[i], linewidth=2)
        ax[i].set_title(metrics[i])
        ax[i].set_xlim(0, np.max(epoch_list))
        # ax[i].legend()
    ax[2].hlines(best_acc, 0, np.max(epoch_list), colors='red', linestyles='dashed')
    ax[2].plot(best_epoch, best_acc, 'ro')
    ax[2].annotate(f'({best_epoch}, {best_acc:.2f}%)', xy=(best_epoch, best_acc), xytext=(-30, -15), textcoords='offset points', color='red')

    result_dir = result_file.rsplit('/', 1)[0]
    fig.savefig(f'{result_dir}/results.png', dpi=300)
    plt.close(fig)


def plot_results_2x2(result_file):
    fig, ax = plt.subplots(2, 2, figsize=(12, 9), tight_layout=True)
    ax = ax.ravel()
    metrics = ['Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy']
    with open(result_file, 'r') as f:
        lines = f.readlines()
    epoch_list = []
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    best_epoch, best_acc = 0, 0.0
    for line in lines:
        if not line.startswith('epoch'):
            continue
        line = line.strip()
        epoch, train_loss, train_acc, test_loss, test_acc = line.split(' | ')[:5]
        epoch_list.append(int(epoch.split(': ')[1]))
        train_loss_list.append(float(train_loss.split(': ')[1]))
        train_acc_list.append(float(train_acc.split(': ')[1]))
        test_loss_list.append(float(test_loss.split(': ')[1]))
        test_acc_list.append(float(test_acc.split(': ')[1]))
        best_accuracy_epoch = line.split(' | ')[-1]
        best_acc = float(best_accuracy_epoch.split(' @ ')[0].split(': ')[1])
        best_epoch = int(best_accuracy_epoch.split(' @ ')[1].split(': ')[1])
    results = [
        np.array(train_loss_list),
        np.array(train_acc_list),
        np.array(test_loss_list),
        np.array(test_acc_list)
    ]

    for i in range(len(metrics)):
        ax[i].plot(epoch_list, results[i], '-', label=metrics[i], linewidth=2)
        ax[i].set_title(metrics[i])
        ax[i].set_xlim(0, np.max(epoch_list))
        # ax[i].legend()
    ax[-1].hlines(best_acc, 0, np.max(epoch_list), colors='red', linestyles='dashed')
    ax[-1].plot(best_epoch, best_acc, 'ro')
    ax[-1].annotate(f'({best_epoch}, {best_acc:.2f}%)', xy=(best_epoch, best_acc), xytext=(-30, -15), textcoords='offset points', color='red')

    result_dir = result_file.rsplit('/', 1)[0]
    fig.savefig(f'{result_dir}/results.png', dpi=300)
    plt.close(fig)


def plot_results_cotraining(result_file):
    fig, ax = plt.subplots(2, 3, figsize=(14, 7), tight_layout=True)
    ax = ax.ravel()
    metrics = ['Net1 Train Loss', 'Net1 Train Accuracy', 'Net1 Test Accuracy',
               'Net2 Train Loss', 'Net2 Train Accuracy', 'Net2 Test Accuracy']
    with open(result_file, 'r') as f:
        lines = f.readlines()
    epoch_list = []
    train_loss_1_list = []
    train_loss_2_list = []
    train_acc_1_list = []
    train_acc_2_list = []
    test_acc_1_list = []
    test_acc_2_list = []
    best_epoch_1, best_acc_1 = 0, 0.0
    best_epoch_2, best_acc_2 = 0, 0.0
    for line in lines:
        if not line.startswith('epoch'):
            continue
        line = line.strip()
        epoch, train_loss, train_acc, test_acc = line.split(' | ')[:4]
        epoch_list.append(int(epoch.split(': ')[1]))
        train_loss1, train_loss2 = map(lambda x: float(x), train_loss.split(': ')[1].lstrip('(').rstrip(')').split('/'))
        train_acc1, train_acc2 = map(lambda x: float(x), train_acc.split(': ')[1].lstrip('(').rstrip(')').split('/'))
        test_acc1, test_acc2 = map(lambda x: float(x), test_acc.split(': ')[1].lstrip('(').rstrip(')').split('/'))
        train_loss_1_list.append(train_loss1)
        train_loss_2_list.append(train_loss2)
        train_acc_1_list.append(train_acc1)
        train_acc_2_list.append(train_acc2)
        test_acc_1_list.append(test_acc1)
        test_acc_2_list.append(test_acc2)

        best_accuracy_epoch = line.split(' | ')[-1]
        best_acc_1, best_acc_2 = map(lambda x: float(x), best_accuracy_epoch.split(' @ ')[0].split(': ')[1].lstrip('(').rstrip(')').split('/'))
        best_epoch_1, best_epoch_2 = map(lambda x: int(x), best_accuracy_epoch.split(' @ ')[1].split(': ')[1].lstrip('(').rstrip(')').split('/'))
    results = [
        np.array(train_loss_1_list),
        np.array(train_acc_1_list),
        np.array(test_acc_1_list),
        np.array(train_loss_2_list),
        np.array(train_acc_2_list),
        np.array(test_acc_2_list)
    ]

    for i in range(len(metrics)):
        ax[i].plot(epoch_list, results[i], '-', label=metrics[i], linewidth=2)
        ax[i].set_title(metrics[i])
        ax[i].set_xlim(0, np.max(epoch_list))
        # ax[i].legend()
    ax[2].hlines(best_acc_1, 0, best_epoch_1, colors='red', linestyles='dashed')
    ax[2].plot(best_epoch_1, best_acc_1, 'ro')
    ax[2].annotate(f'({best_epoch_1}, {best_acc_1:.2f}%)', xy=(best_epoch_1, best_acc_1), xytext=(-30, -15), textcoords='offset points', color='red')
    ax[5].hlines(best_acc_2, 0, best_epoch_2, colors='red', linestyles='dashed')
    ax[5].plot(best_epoch_2, best_acc_2, 'ro')
    ax[5].annotate(f'({best_epoch_2}, {best_acc_2:.2f}%)', xy=(best_epoch_2, best_acc_2), xytext=(-30, -15), textcoords='offset points', color='red')

    result_dir = result_file.rsplit('/', 1)[0]
    fig.savefig(f'{result_dir}/results.png', dpi=300)
    plt.close(fig)



if __name__ == '__main__':
    logfile_path = '../Results/20200622_113934-cifar100-clean_openset0.2_baseline-resnet18-bestAcc_81.0125/log.txt'
    plot_results(logfile_path)
    # plt.show()
