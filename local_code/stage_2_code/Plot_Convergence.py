'''
Training convergence plotting utility.

Produces the "loss vs. training epoch" figure required by report section 3.5,
showing that the model converges gradually under gradient descent.
'''

import os

import matplotlib
# Use a non-interactive backend so the figure can be written to disk on any
# machine, including headless ones (CI, remote shells, graders' containers).
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_training_convergence(loss_history,
                              accuracy_history=None,
                              save_path=None,
                              title='MLP Training Convergence'):
    '''
    Plot training loss against epoch and save the figure to disk.

    Parameters
    ----------
    loss_history : list of float
        Training loss recorded once per epoch. Index i is epoch i.
    accuracy_history : list of float, optional
        Training accuracy recorded once per epoch. When supplied it is drawn
        on a secondary y-axis for context.
    save_path : str, optional
        Destination file path for the figure (e.g. '.../convergence.png').
        Parent directories are created if they do not exist.
    title : str
        Figure title.

    Returns
    -------
    str or None
        The path the figure was written to, or None if it was not saved.
    '''
    if not loss_history:
        raise ValueError(
            'loss_history is empty - nothing to plot. Make sure the model '
            'recorded a loss value for each training epoch.'
        )

    epochs = range(len(loss_history))

    fig, ax_loss = plt.subplots(figsize=(8, 5))

    # ---- primary axis: training loss (the quantity section 3.5 asks for) ----
    ax_loss.plot(epochs, loss_history,
                 color='tab:red', linewidth=2, label='Training loss')
    ax_loss.set_xlabel('Training epoch')
    ax_loss.set_ylabel('Cross-entropy loss', color='tab:red')
    ax_loss.tick_params(axis='y', labelcolor='tab:red')
    ax_loss.grid(True, alpha=0.3)

    handles, labels = ax_loss.get_legend_handles_labels()

    # ---- optional secondary axis: training accuracy ----
    if accuracy_history:
        ax_acc = ax_loss.twinx()
        ax_acc.plot(range(len(accuracy_history)), accuracy_history,
                    color='tab:blue', linewidth=2,
                    linestyle='--', label='Training accuracy')
        ax_acc.set_ylabel('Training accuracy', color='tab:blue')
        ax_acc.tick_params(axis='y', labelcolor='tab:blue')
        ax_acc.set_ylim(0, 1)

        acc_handles, acc_labels = ax_acc.get_legend_handles_labels()
        handles += acc_handles
        labels += acc_labels

    ax_loss.set_title(title)
    ax_loss.legend(handles, labels, loc='center right')
    fig.tight_layout()

    saved_to = None
    if save_path:
        directory = os.path.dirname(os.path.abspath(save_path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        saved_to = save_path
        print('convergence plot saved to:', save_path)

    plt.close(fig)
    return saved_to


def plot_kfold_convergence(fold_loss_histories,
                           save_path=None,
                           title='MLP Training Convergence per Fold'):
    '''
    Plot one training-loss curve per cross-validation fold.

    Parameters
    ----------
    fold_loss_histories : list of list of float
        One per-epoch loss curve per fold.
    save_path : str, optional
        Destination file path for the figure. Parent directories are created
        if they do not exist.
    title : str
        Figure title.

    Returns
    -------
    str or None
        The path the figure was written to, or None if it was not saved.
    '''
    if not fold_loss_histories:
        raise ValueError(
            'fold_loss_histories is empty - nothing to plot. Only '
            'gradient-based methods record a loss curve.'
        )

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, history in enumerate(fold_loss_histories, start=1):
        ax.plot(range(len(history)), history,
                linewidth=2, label='Fold {}'.format(i))

    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Cross-entropy loss')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    fig.tight_layout()

    saved_to = None
    if save_path:
        directory = os.path.dirname(os.path.abspath(save_path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        saved_to = save_path
        print('k-fold convergence plot saved to:', save_path)

    plt.close(fig)
    return saved_to
