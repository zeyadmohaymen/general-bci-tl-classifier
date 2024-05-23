import pickle
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.pipeline import Pipeline
from scripts.epochs_preprocessing import EventsEncoder, EventsEqualizer, Cropper, EpochsSegmenter
from scripts.label_alignment import LabelAlignment
from scripts.ts_feature_extraction import TangentSpaceMapping
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np

def create_preprocessing_pipeline():
    return Pipeline([
        ('encoder', EventsEncoder()),
        ('equalizer', EventsEqualizer()),
        ('cropper', Cropper()),
        ('segmenter', EpochsSegmenter())
    ])


def create_preprocessing_grid():
    return ParameterGrid([
        {
            'encoder__counter_class': ['rest', 'other'],
            'cropper__tmax': [2.5, 3.5],
            'segmenter': ['passthrough']
        },
        {
            'encoder__counter_class': ['rest', 'other'],
            'cropper__tmax': [2.5],
            'segmenter__window_size': [1],
            'segmenter__overlap': [0.2, 0.5]
        },
        {
            'encoder__counter_class': ['rest', 'other'],
            'cropper__tmax': [3.5],
            'segmenter__window_size': [1, 2],
            'segmenter__overlap': [0.2, 0.5]
        }
    ])


def create_classifier_pipeline():
    return Pipeline([
        ('tsm', TangentSpaceMapping()),
        ('clf', 'passthrough')
    ])


def create_classifier_grid():
    return ParameterGrid({
        'clf': [LinearDiscriminantAnalysis(), SVC(kernel='linear'), LogisticRegression(class_weight='balanced')]
    })

def grid_search_cv(pipe, pipe_grid, clf, clf_grid, subject_epochs, cv):
    results = pd.DataFrame(columns=['Counter class', 'Epoch duration (s)', 'Segmentation', 'Window size (s)', 'Overlap (%)', 'Classifier', 'Accuracy', 'F1 score'])

    #* Each preprocessing parameter combination
    for pipe_params in pipe_grid:
        pipe.set_params(**pipe_params)
        preprocessed_epochs = pipe.fit_transform(subject_epochs)    #? mne.Epochs --> mne.Epochs

        #* Each classifier
        for clf_params in clf_grid:
            clf.set_params(**clf_params)

            #* Each fold
            fold_results = _kfold_cv_with_label_alignment(preprocessed_epochs, clf, cv=cv)

            results = results.append({
                'Counter class': 'feet vs rest' if pipe_params['encoder__counter_class'] == 'rest' else 'feet vs no feet',
                'Epoch duration (s)': pipe_params['cropper__tmax'] - 0.5,
                'Segmentation': 'No' if pipe_params.get('segmenter', None) == 'passthrough' else 'Yes',
                'Window size (s)': pipe_params.get('segmenter__window_size', '-'),
                'Overlap (%)': pipe_params.get('segmenter__overlap', '-'),
                'Classifier': clf_params['clf'].__class__.__name__,
                'Accuracy': np.mean(fold_results['acc']),
                'F1 score': np.mean(fold_results['f1'])
            }, ignore_index=True)

    return results

def kfold_cv_with_label_alignment(subject_epochs, clf, cv):
    fold_results = dict(acc=[], f1=[])
    for train_idx, test_idx in cv.split(subject_epochs):

        train_epochs = subject_epochs[train_idx]    #! 80% of total epochs
        test_epochs = subject_epochs[test_idx]      #! 20% of total epochs

        # Split train epochs source and calibration
        source_epochs, calibration_epochs = _la_split(train_epochs)

        # Label alignment
        la = LabelAlignment(target_epochs=calibration_epochs)
        aligned_data, aligned_events = la.fit_transform(source_epochs)    #? mne.Epochs --> np.ndarray, np.ndarray

        # Classifier
        clf.fit(aligned_data, aligned_events)

        # Scoring
        test_data = test_epochs.get_data(copy=True)
        test_events = test_epochs.events[:, -1]
        y_pred = clf.predict(test_data)
        y_true = test_events

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        fold_results['acc'].append(accuracy)
        fold_results['f1'].append(f1)

    return fold_results

def la_split(train_epochs):
    # Split train epochs source and calibration
    source_epochs = train_epochs[:int(0.875*len(train_epochs))]         #! 70% of total epochs
    calibration_epochs = train_epochs[int(0.875*len(train_epochs)):]    #! 10% of total epochs

    return source_epochs, calibration_epochs

# Load the data in pickled_data folder into a dictionary
def load_data():
    data = {
        'physionet': pickle.load(open('pickled_data/physionet.pkl', 'rb')),
        'schirrmeister': pickle.load(open('pickled_data/schirrmeister.pkl', 'rb')),
        'weibo': pickle.load(open('pickled_data/weibo.pkl', 'rb'))
    }
    return data