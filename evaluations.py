import pickle
from sklearn.model_selection import KFold, LeaveOneGroupOut
from scripts.evals_util import load_data, create_preprocessing_pipeline, create_preprocessing_grid, create_classifier_pipeline, create_classifier_grid, grid_search_cv_la, svm_pipeline, svm_grid
from matplotlib.backends.backend_pdf import PdfPages
from mne import concatenate_epochs, set_log_level
import matplotlib.pyplot as plt
import numpy as np

def same_subject_evaluation(data):
    # Preprocessing pipeline
    pipe = create_preprocessing_pipeline()
    pipe_grid = create_preprocessing_grid()

    # Classifier pipeline
    clf = svm_pipeline()
    clf_grid = svm_grid()

    results = {}
    params_accs = {}

    # Each dataset
    for dataset_name, dataset in data.items():
        results[dataset_name] = {}
        
        params_accs[dataset_name] = {}
        subj_ids = [subject_id for subject_id in dataset.keys()]
        
        #* Each subject
        for subject_id, subject_epochs in dataset.items():
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            results[dataset_name][subject_id], _, params_acc_dict = grid_search_cv_la(pipe=pipe, pipe_grid=pipe_grid, clf=clf, clf_grid=clf_grid, epochs=subject_epochs, cv=cv, la_split_ratio=0.875, calib_from="train", acc_dict=True)

            # Sort the results by accuracy
            results[dataset_name][subject_id].sort_values(by='Accuracy', ascending=False, inplace=True)

            # Save the results to a CSV file
            results[dataset_name][subject_id].to_csv(f'results/same_subject/{dataset_name}/{subject_id}.csv', index=False)

            for param, acc in params_acc_dict.items():
                if param not in params_accs[dataset_name]:
                    params_accs[dataset_name][param] = []
                params_accs[dataset_name][param].append(acc)

        # Bar chart
        bars = []
        for param, accs in params_accs[dataset_name].items():
            accs = np.array(accs) * 100
            fig, ax = plt.subplots()
            ax.bar(subj_ids, accs)
            ax.set_xlabel('Subject ID')
            ax.set_ylabel('Accuracy (%)')
            ax.set_ylim([0, 100])
            ax.set_title(param)
            fig.tight_layout()
            bars.append(fig)
            plt.close(fig)

        # Save the bar charts to a PDF file
        with PdfPages(f'results/same_subject/{dataset_name}/bars.pdf') as pdf:
            for fig in bars:
                pdf.savefig(fig)

    # Save the results to a pickle file
    with open('results/same_subject/results.pkl', 'wb') as f:
        pickle.dump(results, f)

    return results

def cross_subject_evaluation(data):
    # Preprocessing pipeline
    pipe = create_preprocessing_pipeline()
    pipe_grid = create_preprocessing_grid()

    # Classifier pipeline
    clf = svm_pipeline()
    clf_grid = svm_grid()

    results = {}

    # Each dataset
    for dataset_name, dataset in data.items():
        results[dataset_name] = {}

        subjects = [subject_epochs for subject_epochs in dataset.values()]

        cv = LeaveOneGroupOut()
        results[dataset_name], figs = grid_search_cv_la(pipe=pipe, pipe_grid=pipe_grid, clf=clf, clf_grid=clf_grid, epochs=subjects, cv=cv, la_split_ratio=0.8, calib_from="test", roc=True)

        # Sort the results by accuracy
        results[dataset_name].sort_values(by='Accuracy', ascending=False, inplace=True)

        # Save the results to a CSV file
        results[dataset_name].to_csv(f'results/cross_subject/{dataset_name}.csv', index=False)

        # Save the ROC curves to a PDF file
        with PdfPages(f'results/cross_subject/{dataset_name}_roc.pdf') as pdf:
            for fig in figs.values():
                pdf.savefig(fig)

    # Save the results to a pickle file
    with open('results/cross_subject/results.pkl', 'wb') as f:
        pickle.dump(results, f)

    return results

def cross_dataset_evaluation(data):
    # Preprocessing pipeline
    pipe = create_preprocessing_pipeline()
    pipe_grid = create_preprocessing_grid()

    # Classifier pipeline
    clf = svm_pipeline()
    clf_grid = svm_grid()

    results = {}

    datasets = [concatenate_epochs([subject_epochs for subject_epochs in dataset.values()]) for dataset in data.values()]

    cv = LeaveOneGroupOut()
    results, figs = grid_search_cv_la(pipe=pipe, pipe_grid=pipe_grid, clf=clf, clf_grid=clf_grid, epochs=datasets, cv=cv, la_split_ratio=0.8, calib_from="test", roc=True, avg_splits=False)

    for dataset_name, dataset_results, fig in zip(data.keys(), results, figs):
        # Sort the results by accuracy
        dataset_results.sort_values(by='Accuracy', ascending=False, inplace=True)

        # Save the results to a CSV file
        dataset_results.to_csv(f'results/cross_dataset/{dataset_name}.csv', index=False)

        # Save the ROC curves to a PDF file
        with PdfPages(f'results/cross_dataset/{dataset_name}_roc.pdf') as pdf:
            for f in fig.values():
                pdf.savefig(f)

    # Save the results to a pickle file
    with open('results/cross_dataset/results.pkl', 'wb') as f:
        pickle.dump(results, f)

    return results

# Main
if __name__ == '__main__':
    set_log_level(verbose='CRITICAL')

    data = load_data()

    print('Starting same-subject evaluation...')
    same_subject_results = same_subject_evaluation(data.copy())
    print('Starting cross-subject evaluation...')
    cross_subject_results = cross_subject_evaluation(data.copy())
    print('Starting cross-dataset evaluation...')
    cross_dataset_results = cross_dataset_evaluation(data.copy())

    print('Done!')    