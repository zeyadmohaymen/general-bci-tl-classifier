import pickle
from sklearn.model_selection import ParameterGrid, KFold
from scripts.evals_util import load_data, create_preprocessing_pipeline, create_preprocessing_grid, create_classifier_pipeline, create_classifier_grid, grid_search_cv


def same_subject_evaluation(data):
    # Preprocessing pipeline
    pipe = create_preprocessing_pipeline()
    pipe_grid = create_preprocessing_grid()

    # Classifier pipeline
    clf = create_classifier_pipeline()
    clf_grid = create_classifier_grid()

    results = {}

    # Each dataset
    for dataset_name, dataset in data.items():
        results[dataset_name] = {}
        
        #* Each subject
        for subject_id, subject_epochs in dataset.items():
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            results[dataset_name][subject_id] = grid_search_cv(pipe, pipe_grid, clf, clf_grid, subject_epochs, cv=cv)

            # Sort the results by accuracy
            results[dataset_name][subject_id].sort_values(by='Accuracy', ascending=False, inplace=True)

            # Save the results to a CSV file
            results[dataset_name][subject_id].to_csv(f'results/same_subject/{dataset_name}/{subject_id}.csv', index=False)

    # Save the results to a pickle file
    with open('results/same_subject/results.pkl', 'wb') as f:
        pickle.dump(results, f)

    return results

# def cross_subject_evaluation(data):
#     # Preprocessing pipeline
#     pipe = create_preprocessing_pipeline()
#     pipe_grid = create_preprocessing_grid()

#     # Classifier pipeline
#     clf = create_classifier_pipeline()
#     clf_grid = create_classifier_grid()

#     results = {}

#     # Each dataset
#     for dataset_name, dataset in data.items():
#         results[dataset_name] = {}
        
#         #* Each subject
#         for subject_id, subject_epochs in dataset.items():
#             cv = KFold(n_splits=5, shuffle=True, random_state=42)
#             results[dataset_name][subject_id] = grid_search_cv(pipe, pipe_grid, clf, clf_grid, subject_epochs, cv=cv)

#             # Sort the results by accuracy
#             results[dataset_name][subject_id].sort_values(by='Accuracy', ascending=False, inplace=True)

#             # Save the results to a CSV file
#             results[dataset_name][subject_id].to_csv(f'results/cross_subject/{dataset_name}/{subject_id}.csv', index=False)

#     # Save the results to a pickle file
#     with open('results/cross_subject/results.pkl', 'wb') as f:
#         pickle.dump(results, f)

#     return results

# Main
if __name__ == '__main__':
    data = load_data()
    results = same_subject_evaluation(data)
    