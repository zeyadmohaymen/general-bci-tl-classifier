import pickle
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from scripts.epochs_preprocessing import EventsEncoder, EventsEqualizer, Cropper, EpochsSegmenter, EpochsDecoder
from scripts.label_alignment import LabelAlignment
from scripts.ts_feature_extraction import TangentSpaceMapping
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_preprocessing_pipeline():
    return Pipeline([
        ('encoder', EventsEncoder()),
        ('equalizer', EventsEqualizer()),
        ('cropper', Cropper()),
        ('segmenter', 'passthrough'),
        ('decoder', EpochsDecoder())
    ])


def create_preprocessing_grid():
    return ParameterGrid([
        {
            'encoder__counter_class': ['rest', 'other'],
            'cropper__tmax': [2.5, 3.5],
        },
        {
            'encoder__counter_class': ['rest', 'other'],
            'cropper__tmax': [2.5],
            'segmenter': [EpochsSegmenter()],
            'segmenter__window_size': [1],
            'segmenter__overlap': [0.2, 0.5]
        },
        {
            'encoder__counter_class': ['rest', 'other'],
            'cropper__tmax': [3.5],
            'segmenter': [EpochsSegmenter()],
            'segmenter__window_size': [1, 2],
            'segmenter__overlap': [0.2, 0.5]
        }
    ])


def create_classifier_pipeline():
    return Pipeline([
        ('tsm', TangentSpaceMapping()),
        ('scaler', StandardScaler()),
        ('clf', 'passthrough')
    ])


def create_classifier_grid():
    return ParameterGrid({
        'clf': [LinearDiscriminantAnalysis(), SVC(kernel='linear'), LogisticRegression(class_weight='balanced')]
    })

def grid_search_cv_la(pipe, pipe_grid, clf, clf_grid, epochs, cv, la_split_ratio, calib_from="test", roc=False, avg_splits=True):
    results = pd.DataFrame(columns=['Counter class', 'Epoch duration (s)', 'Segmentation', 'Window size (s)', 'Overlap (%)', 'Classifier', 'Accuracy', 'F1 score', 'Train duration (s)', 'Calibration duration (s)', 'Test duration (s)'])
    figs = {}
    cv_groups = None

    #* Each preprocessing parameter combination
    for pipe_params in pipe_grid:
        pipe.set_params(**pipe_params)
        if isinstance(epochs, list):
            data, ev = zip(*[pipe.fit_transform(subject_epochs) for subject_epochs in epochs])
            preprocessed_data = np.concatenate(data)
            preprocessed_events = np.concatenate(ev)[:,-1]
            cv_groups = np.concatenate([[i] * len(subject_data) for i, subject_data in enumerate(data)])
        else:
            preprocessed_data, preprocessed_events = pipe.fit_transform(epochs)
            preprocessed_events = preprocessed_events[:, -1]
        
        # preprocessed_epochs.reset_drop_log_selection()

        #* Each classifier
        for clf_params in clf_grid:
            clf.set_params(**clf_params)

            #* Each fold
            cv_results = _cv_with_label_alignment(preprocessed_data, preprocessed_events, clf=clf, cv=cv, cv_groups=cv_groups, la_split_ratio=la_split_ratio, calib_from=calib_from, roc=roc)

            if avg_splits == False:
                results = []
                figs = []
                for i in range(len(cv_results['acc'])):
                    df = pd.DataFrame(columns=['Counter class', 'Epoch duration (s)', 'Segmentation', 'Window size (s)', 'Overlap (%)', 'Classifier', 'Accuracy', 'F1 score', 'Train duration (s)', 'Calibration duration (s)', 'Test duration (s)'])
                    df = df.append({
                        'Counter class': 'feet vs rest' if pipe_params['encoder__counter_class'] == 'rest' else 'feet vs no feet',
                        'Epoch duration (s)': pipe_params['cropper__tmax'] - 0.5,
                        'Segmentation': 'No' if pipe_params.get('segmenter', None) == 'passthrough' else 'Yes',
                        'Window size (s)': pipe_params.get('segmenter__window_size', '-'),
                        'Overlap (%)': pipe_params.get('segmenter__overlap', '-'),
                        'Classifier': clf_params['clf'].__class__.__name__,
                        'Accuracy': cv_results['acc'][i],
                        'F1 score': cv_results['f1'][i],
                        'Train duration (s)': cv_results['durations']['train'][i] * pipe_params['cropper__tmax'] - 0.5,
                        'Calibration duration (s)': cv_results['durations']['calib'][i] * pipe_params['cropper__tmax'] - 0.5,
                        'Test duration (s)': cv_results['durations']['test'][i] * pipe_params['cropper__tmax'] - 0.5
                    }, ignore_index=True)
                    results.append(df)

                    if roc:
                        fig = RocCurveDisplay(fpr=cv_results['fprs'][i], tpr=cv_results['tprs'][i]).figure_
                        fig.suptitle(f"{results.iloc[-1]['Counter class']} - {results.iloc[-1]['Epoch duration (s)']} - {results.iloc[-1]['Segmentation']} - {results.iloc[-1]['Classifier']}")
                        figs.append(dict(f"{pipe_params['encoder__counter_class']}_{pipe_params['cropper__tmax']-0.5}_{pipe_params.get('segmenter', 'No')}_{'-' if pipe_params.get('segmenter', 'No') == 'No' else pipe_params.get('segmenter__window_size', '-')}_{pipe_params.get('segmenter__overlap', '-')}_{clf_params['clf'].__class__.__name__}"), fig)
            else:

                results = results.append({
                    'Counter class': 'feet vs rest' if pipe_params['encoder__counter_class'] == 'rest' else 'feet vs no feet',
                    'Epoch duration (s)': pipe_params['cropper__tmax'] - 0.5,
                    'Segmentation': 'No' if pipe_params.get('segmenter', None) == 'passthrough' else 'Yes',
                    'Window size (s)': pipe_params.get('segmenter__window_size', '-'),
                    'Overlap (%)': pipe_params.get('segmenter__overlap', '-'),
                    'Classifier': clf_params['clf'].__class__.__name__,
                    'Accuracy': np.mean(cv_results['acc']),
                    'F1 score': np.mean(cv_results['f1']),
                    'Train duration (s)': np.mean(np.array(cv_results['durations']['train']) * pipe_params['cropper__tmax'] - 0.5),
                    'Calibration duration (s)': np.mean(np.array(cv_results['durations']['calib']) * pipe_params['cropper__tmax'] - 0.5),
                    'Test duration (s)': np.mean(np.array(cv_results['durations']['test']) * pipe_params['cropper__tmax'] - 0.5)
                }, ignore_index=True)

                if roc:
                    fig, ax = create_roc_plot(cv_results['fprs'], cv_results['tprs'])
                    fig.suptitle(f"{results.iloc[-1]['Counter class']} - {results.iloc[-1]['Epoch duration (s)']} - {results.iloc[-1]['Segmentation']} - {results.iloc[-1]['Classifier']}")
                    figs[f"{pipe_params['encoder__counter_class']}_{pipe_params['cropper__tmax']-0.5}_{pipe_params.get('segmenter', 'No')}_{'-' if pipe_params.get('segmenter', 'No') == 'No' else pipe_params.get('segmenter__window_size', '-')}_{pipe_params.get('segmenter__overlap', '-')}_{clf_params['clf'].__class__.__name__}"] = fig

    return results, figs

def _cv_with_label_alignment(data, events, clf, cv, la_split_ratio, cv_groups=None, calib_from="test", roc=False):
    # Validate calibration source
    if calib_from not in ["train", "test"]:
        raise ValueError("calib_from must be either 'train' or 'test'")

    split_results = dict(acc=[], f1=[], durations=dict(train=[], calib=[], test=[]), fprs=[], tprs=[])
    for train_idx, test_idx in cv.split(data, groups=cv_groups):
        train_data, train_events = data[train_idx], events[train_idx]
        test_data, test_events = data[test_idx], events[test_idx]

        # Split train epochs source and calibration
        if calib_from == "train":
            train_data, train_events, calibration_data, calibration_events = _equalized_la_split(train_data, train_events, la_split_ratio)
        else:
            calibration_data, calibration_events, test_data, test_events = _equalized_la_split(train_data, train_events, 1-la_split_ratio)

        # Label alignment
        la = LabelAlignment(target_data=calibration_data, target_events=calibration_events)
        aligned_data, aligned_events = la.fit_transform(train_data, train_events)

        # Classifier
        clf.fit(aligned_data, aligned_events)

        y_pred = clf.predict(test_data)
        y_true = test_events

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        split_results['acc'].append(accuracy)
        split_results['f1'].append(f1)
        split_results['durations']['train'].append(_epochs_duration(train_data))
        split_results['durations']['calib'].append(_epochs_duration(calibration_data))
        split_results['durations']['test'].append(_epochs_duration(test_data))

        if roc:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            split_results['fprs'].append(fpr)
            split_results['tprs'].append(tpr)

    return split_results

# def kfold_cv_with_label_alignment(subject_epochs, clf, cv):
#     fold_results = dict(acc=[], f1=[])
#     for train_idx, test_idx in cv.split(subject_epochs):

#         train_epochs = subject_epochs[train_idx]    #! 80% of total epochs
#         test_epochs = subject_epochs[test_idx]      #! 20% of total epochs

#         # Split train epochs source and calibration
#         train_epochs, calibration_epochs = equalized_la_split(train_epochs, 0.875)

#         # Label alignment
#         la = LabelAlignment(target_epochs=calibration_epochs)
#         aligned_data, aligned_events = la.fit_transform(train_epochs)    #? mne.Epochs --> np.ndarray, np.ndarray

#         # Classifier
#         clf.fit(aligned_data, aligned_events)

#         # Scoring
#         test_data = test_epochs.get_data(copy=True)
#         test_events = test_epochs.events[:, -1]
#         y_pred = clf.predict(test_data)
#         y_true = test_events

#         accuracy = accuracy_score(y_true, y_pred)
#         f1 = f1_score(y_true, y_pred)

#         fold_results['acc'].append(accuracy)
#         fold_results['f1'].append(f1)

#     return fold_results

# def loo_cv_with_label_alignment(dataset_subjects, clf):
#     # Convert subjects dictionary to array of epochs
#     subjects = [subject_epochs for subject_epochs in dataset_subjects.values()]

#     loo = LeaveOneOut()
    
#     split_results = dict(acc=[], f1=[])
#     for train_idx, test_idx in loo.split(subjects):
#         train_epochs = concatenate_epochs([subjects[idx] for idx in train_idx])
#         test_epochs = subjects[test_idx[0]]

#         # Split train epochs source and calibration
#         calibration_epochs, test_epochs = equalized_la_split(test_epochs, 0.2)

#         # Label alignment
#         la = LabelAlignment(target_epochs=calibration_epochs)
#         aligned_data, aligned_events = la.fit_transform(train_epochs)

#         # Classifier
#         clf.fit(aligned_data, aligned_events)

#         # Scoring
#         test_data = test_epochs.get_data(copy=True)
#         test_events = test_epochs.events[:, -1]
#         y_pred = clf.predict(test_data)
#         y_true = test_events

#         accuracy = accuracy_score(y_true, y_pred)
#         f1 = f1_score(y_true, y_pred)

#         split_results['acc'].append(accuracy)
#         split_results['f1'].append(f1)

def _epochs_duration(epochs, tmin=0, tmax=1):
    return len(epochs)*(tmax - tmin)

def _equalized_la_split(data, events, split_ratio, eq=True, classes=[0, 1]):
    if eq:
        # Split data and events by class
        class_data = {class_: data[events == class_] for class_ in classes}
        class_events = {class_: events[events == class_] for class_ in classes}

        # Split each class data and events
        split_1_data = {class_: class_data[class_][:int(split_ratio*len(class_data[class_]))] for class_ in classes}
        split_1_events = {class_: class_events[class_][:int(split_ratio*len(class_events[class_]))] for class_ in classes}
        split_2_data = {class_: class_data[class_][int(split_ratio*len(class_data[class_])):] for class_ in classes}
        split_2_events = {class_: class_events[class_][int(split_ratio*len(class_events[class_])):] for class_ in classes}

        # Concatenate the split data and events
        split_1_data = np.concatenate([split_1_data[class_] for class_ in classes])
        split_1_events = np.concatenate([split_1_events[class_] for class_ in classes])
        split_2_data = np.concatenate([split_2_data[class_] for class_ in classes])
        split_2_events = np.concatenate([split_2_events[class_] for class_ in classes])

        return split_1_data, split_1_events, split_2_data, split_2_events
    
    else:
        split_1_data = data[:int(split_ratio*len(data))]
        split_1_events = events[:int(split_ratio*len(events))]
        split_2_data = data[int(split_ratio*len(data)):]
        split_2_events = events[int(split_ratio*len(events)):]

        return split_1_data, split_1_events, split_2_data, split_2_events
    
def create_roc_plot(fprs, tprs, color='b', fig=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        
    mean_fpr = np.linspace(0, 1, 100)
    interp_tprs = []
    aucs = []

    for fpr, tpr in zip(fprs, tprs):
        ax.plot(fpr, tpr, color, alpha=0.15, lw=0.5)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))

    interp_tprs = np.array(interp_tprs)
    mean_tpr = interp_tprs.mean(axis=0)
    std_tpr = interp_tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color=color, label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc))
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=0.1)

    ax.plot([0, 1], [0, 1],'r--')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    return fig, ax

# Load the data in pickled_data folder into a dictionary
def load_data():
    data = {
        'physionet': pickle.load(open('pickled_data/physionet.pkl', 'rb')),
        'schirrmeister': pickle.load(open('pickled_data/schirrmeister.pkl', 'rb')),
        'weibo': pickle.load(open('pickled_data/weibo.pkl', 'rb'))
    }
    return data