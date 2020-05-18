# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from keras.wrappers.scikit_learn import KerasClassifier 
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing   import LabelEncoder
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline

import numpy as np
import pandas as pd

from symbols import DATAPATH, MODELPATH
from util import get_starttime, calc_runtime


#
# Chapter 6: Develop your first neural network with Keras
#
def first_nn():
    # load the dataset
    fnm = f'{DATAPATH}pima-indians-diabetes.csv'
    dataset = loadtxt(fnm, delimiter=',')

    # split into input (X) and output (y) variables
    X = dataset[:,0:8]
    y = dataset[:,8]

    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

    # fit the keras model on the dataset
    model.fit(X, y, epochs=150, batch_size=16)

    # evaluate the keras model
    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy*100))

#
# Chapter 7: Evaluate the performance of deep learning models
#
def keras_auto_cv():
    #load pima indians dataset
    fnm=f'{DATAPATH}pima-indians-diabetes.csv'
    dataset = np.loadtxt(fnm, delimiter=",") 

    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]
    
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    
    # Fit the model
    model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10)

def keras_manual_cv():
    # load pima indians dataset
    fnm=f'{DATAPATH}pima-indians-diabetes.csv'
    dataset = np.loadtxt(fnm, delimiter=",") 

    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]

    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=True) 
    cvscores = []
    for train, test in kfold.split(X, Y):

        # Create model
        model = Sequential()
        model.add(Dense(12, input_dim=8, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
    
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # Fit the model
        model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
    
        # Evaluate the model
        scores = model.evaluate(X[test], Y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)) 
        cvscores.append(scores[1] * 100)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


#
# Chapter 8: Use Keras Models withg Scikit-Learn for general machine learning
# 
def scikit_auto_cv():

    # Function to create model, required for KerasClassifier
    def create_model():
        # create model
        model = Sequential()
        model.add(Dense(12, input_dim=8, activation='relu')) 
        model.add(Dense(8, activation='relu')) 
        model.add(Dense(1, activation='sigmoid'))

        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
        return model

    # load pima indians dataset
    fnm=f'{DATAPATH}pima-indians-diabetes.csv'
    dataset = np.loadtxt(fnm, delimiter=",") 
    
    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]

    # create model
    model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
    
    # evaluate using 10-fold cross validation
    kfold = StratifiedKFold(n_splits=10, shuffle=True) 
    results = cross_val_score(model, X, Y, cv=kfold) 
    print(results.mean())


def scikit_grid_search():
    # Function to create model, required for KerasClassifier
    def create_model(optimizer='rmsprop', init='glorot_uniform'):
        # create model
        model = Sequential()
        model.add(Dense(12, input_dim=8, kernel_initializer=init, activation='relu')) 
        model.add(Dense(8, kernel_initializer=init, activation='relu'))
        model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
        
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy']) 
        return model

    # load pima indians dataset
    fnm=f'{DATAPATH}pima-indians-diabetes.csv'
    dataset = np.loadtxt(fnm, delimiter=",")  
    
    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]
    
    # create model
    model = KerasClassifier(build_fn=create_model, verbose=0)
    
    # grid search epochs, batch size and optimizer
    optimizers = ['rmsprop', 'adam']
    inits = ['glorot_uniform', 'normal', 'uniform']
    epochs = [50, 100, 150]
    batches = [5, 10, 20]
    param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=inits) 
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_result = grid.fit(X, Y)
    
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

#
# Chapter 9: Project: Multiclass Classification Of Flower Species
#
def iris_multiclass():
    
    # load dataset
    fnm=f'{DATAPATH}iris.csv'
    dataframe = pd.read_csv(fnm)
    cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm" ] 
    target = "Species"
    X = dataframe[cols].astype(float)
    Y = dataframe[target] 
    
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    
    # convert integers to dummy variables (i.e. one hot encoded) 
    dummy_y = np_utils.to_categorical(encoded_Y)

    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(8, input_dim=4, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
        return model
    
    estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0) 
    kfold = KFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, X, dummy_y, cv=kfold)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#
# Chapter 10: Project: Binary Classification Of Sonar Returns
#
def sonar_classification():
    # load dataset
    fnm=f'{DATAPATH}/sonar.csv'
    dataframe = pd.read_csv(fnm, header=None)
    dataset = dataframe.values
    
    # split into input (X) and output (Y) variables
    X = dataset[:,0:60].astype(float)
    Y = dataset[:,60]
    
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    
    # baseline model
    def create_baseline():
        # create model
        model = Sequential()
        model.add(Dense(60, input_dim=60, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
        return model
    
    # evaluate baseline model with standardized dataset
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100,
                        batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)

    print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

def small_sonar_classification():
    # load dataset
    fnm=f'{DATAPATH}/sonar.csv'
    dataframe = pd.read_csv(fnm, header=None)
    dataset = dataframe.values
    
    # split into input (X) and output (Y) variables
    X = dataset[:,0:60].astype(float)
    Y = dataset[:,60]
    
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    
    # baseline model
    def create_baseline():
        # create model
        model = Sequential()
        model.add(Dense(30, input_dim=60, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
        return model
    
    # evaluate baseline model with standardized dataset
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100,
                        batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)

    print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


def large_sonar_classification():
    # load dataset
    fnm=f'{DATAPATH}/sonar.csv'
    dataframe = pd.read_csv(fnm, header=None)
    dataset = dataframe.values
    
    # split into input (X) and output (Y) variables
    X = dataset[:,0:60].astype(float)
    Y = dataset[:,60]
    
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    
    # baseline model
    def create_baseline():
        # create model
        model = Sequential()
        model.add(Dense(60, input_dim=60, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
        return model
    
    # evaluate baseline model with standardized dataset
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100,
                        batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)

    print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

def grid_sonar_classification():
    # load dataset
    fnm=f'{DATAPATH}/sonar.csv'
    dataframe = pd.read_csv(fnm, header=None)
    dataset = dataframe.values
    
    # split into input (X) and output (Y) variables
    X = dataset[:,0:60].astype(float)
    Y = dataset[:,60]
    
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    
    # baseline model
    def create_baseline(optimizer='adam', init='glorot_uniform'):
        # create model
        model = Sequential()
        model.add(Dense(30, input_dim=60, kernel_initializer=init, activation='relu'))
        #model.add(Dense(30, kernel_initializer=init, activation='relu'))
        model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
        
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy']) 
        return model
    
    # evaluate baseline model with standardized dataset
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    # estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100,
    #                     batch_size=5, verbose=0)))
    estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)

    # grid search epochs, batch size and optimizer
    optimizers = ['rmsprop', 'adam']
    inits = ['glorot_uniform', 'normal', 'uniform']
    epochs = [50, 100, 150]
    batches = [1, 5, 10, 20]
    param_grid = dict(mlp__optimizer=optimizers, 
                      mlp__epochs=epochs, 
                      mlp__batch_size=batches, 
                      mlp__init=inits) 
    grid = GridSearchCV(estimator=pipeline, n_jobs=-1, param_grid=param_grid, cv=kfold)
    print(f'pipeline.get_params().keys()={pipeline.get_params().keys()}')
    grid_result = grid.fit(X, Y)
    
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))  

#
# Chapter 11: Project: Regression Of Boston House Prices
#
def boston_base_regression():
    # Regression Example With Boston Dataset: Baseline

    # load dataset
    fnm=f'{DATAPATH}boston.csv'
    dataframe = pd.read_csv(fnm, delim_whitespace=True)
    dataset = dataframe.values

    # split into input (X) and output (Y) variables
    X = dataset[:,0:13]
    Y = dataset[:,13]

    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(13, input_dim=13, activation='relu')) 
        model.add(Dense(1))
        
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam') 
        return model
    
    # evaluate model
    estimator = KerasRegressor(build_fn=baseline_model, 
                               epochs=100, batch_size=5, verbose=0) 
    kfold = KFold(n_splits=10)
    results = cross_val_score(estimator, X, Y, cv=kfold, n_jobs=-1)
    print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

def boston_standardize_regression():
    # load dataset
    fnm=f'{DATAPATH}boston.csv'
    dataframe = pd.read_csv(fnm, delim_whitespace=True)
    dataset = dataframe.values
    
    # split into input (X) and output (Y) variables
    X = dataset[:,0:13]
    Y = dataset[:,13]
    
    # define base model
    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(13, input_dim=13, activation='relu')) 
        model.add(Dense(1))

        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam') 
        return model
    
    # evaluate model with standardized dataset
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, 
                        epochs=100, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10)
    results = cross_val_score(pipeline, X, Y, cv=kfold, n_jobs=-1)
    print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

def boston_standardize_regression_deep():
    # load dataset
    fnm=f'{DATAPATH}boston.csv'
    dataframe = pd.read_csv(fnm, delim_whitespace=True)
    dataset = dataframe.values
    
    # split into input (X) and output (Y) variables
    X = dataset[:,0:13]
    Y = dataset[:,13]

    def larger_model():
        # create model
        model = Sequential()
        model.add(Dense(13, input_dim=13, activation='relu')) 
        model.add(Dense(6, activation='relu'))
        model.add(Dense(1))
        
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam') 
        return model

    # evaluate model with standardized dataset
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=larger_model, 
                       epochs=100, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10)
    results = cross_val_score(pipeline, X, Y, cv=kfold, n_jobs=-1)
    print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))

def boston_standardize_regression_wide():
    # load dataset
    fnm=f'{DATAPATH}boston.csv'
    dataframe = pd.read_csv(fnm, delim_whitespace=True)
    dataset = dataframe.values
    
    # split into input (X) and output (Y) variables
    X = dataset[:,0:13]
    Y = dataset[:,13]

    def wider_model():
        # create model
        model = Sequential()
        model.add(Dense(20, input_dim=13, activation='relu')) 
        # model.add(Dense(6, activation='relu'))
        model.add(Dense(1))
        
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam') 
        return model

    # evaluate model with standardized dataset
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=wider_model, 
                       epochs=50, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10)
    results = cross_val_score(pipeline, X, Y, cv=kfold, n_jobs=-1)
    print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))


def boston_standardize_regression_mixed():
    # load dataset
    fnm=f'{DATAPATH}boston.csv'
    dataframe = pd.read_csv(fnm, delim_whitespace=True)
    dataset = dataframe.values
    
    # split into input (X) and output (Y) variables
    X = dataset[:,0:13]
    Y = dataset[:,13]

    def wider_model():
        # create model
        model = Sequential()
        model.add(Dense(20, input_dim=13, activation='relu'))
        model.add(Dense(6, activation='relu'))
        model.add(Dense(1))
        
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam') 
        return model

    # evaluate model with standardized dataset
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=wider_model, 
                       epochs=50, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10)
    results = cross_val_score(pipeline, X, Y, cv=kfold, n_jobs=-1)
    print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))

#
# Chapter 12: Save Your Models For Later With Serialization
#
def save_model_file():
    #load pima indians dataset
    fnm=f'{DATAPATH}pima-indians-diabetes.csv'
    dataset = np.loadtxt(fnm, delimiter=",") 

    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]
    
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    
    # Fit the model
    model.fit(X, Y, epochs=150, batch_size=10, verbose=0)
    
    # evaluate the model
    scores = model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)) 
    
    # save model and architecture to single file 
    fnm = f'{MODELPATH}model.h5'
    model.save(fnm)
    print("Saved model to disk")


def load_model_file():
    # load model
    fnm = f'{MODELPATH}model.h5'
    model = load_model(fnm) 
    
    # summarize model. 
    model.summary()
    
    # load dataset
    fnm=f'{DATAPATH}pima-indians-diabetes.csv'
    dataset = np.loadtxt(fnm, delimiter=",")
    
    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]
    
    # evaluate the model
    score = model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

#
# Chapter 13: Keep The Best Models During Training With Checkpointing
#

def checkpoint_model_improvements():
    # load pima indians dataset
    fnm=f'{DATAPATH}pima-indians-diabetes.csv'
    dataset = np.loadtxt(fnm, delimiter=",") 

    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]

    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

    # checkpoint
    filepath = f'{MODELPATH}'+'weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                                save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Fit the model
    model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, 
            callbacks=callbacks_list, verbose=0)    

def checkpoint_best_model_only():
    # load pima indians dataset
    fnm=f'{DATAPATH}pima-indians-diabetes.csv'
    dataset = np.loadtxt(fnm, delimiter=",") 

    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]
    
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    
    # checkpoint
    filepath=f'{MODELPATH}'+'weights.best.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='max') 
    callbacks_list = [checkpoint]
        
    # Fit the model
    model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, 
              callbacks=callbacks_list, verbose=0)

def print_runtime(func):
    start = get_starttime()
    func()
    calc_runtime(start, True)

if __name__ == "__main__":
    # funcs_to_run = [first_nn, keras_auto_cv, keras_manual_cv, 
    #                 scikit_auto_cv, scikit_grid_search, 
    #                 iris_multiclass, sonar_classification,
    #                 small_sonar_classification, large_sonar_classification,
    #                 grid_sonar_classification, boston_base_regression,
    #                 boston_standardize_regression, 
    #                 boston_standardize_regression_deep,
    #                 boston_standardize_regression_wide,
    #                 boston_standardize_regression_mixed,
    #                 save_model_file, load_model_file ]
    #
    funcs_to_run = [ checkpoint_model_improvements, 
                     checkpoint_best_model_only
                    ]

    for i, f in enumerate(funcs_to_run):
        stars="*"*80
        print(f'{stars}\n{stars}')
        print(f'run={i} function={f}\n')
        print_runtime(f)
        print(f'{stars}\n{stars}\n')
