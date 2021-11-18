from keras.models import Sequential, Model
from keras.layers import InputLayer, Activation, BatchNormalization, Flatten, Dropout
from keras.layers import Dense, Conv2D, MaxPooling2D, ConvLSTM2D, Lambda
from keras.layers import Conv1D, MaxPooling1D
from keras import regularizers
from keras.layers import Bidirectional, TimeDistributed, concatenate
from keras.backend import reverse
from keras.engine.input_layer import Input

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def make_rf_model(random_state, n_jobs, n_estimators, class_weight):
    """
    Defines a sklearn random forest model. See sci-kit learn
    documentation of sklearn.ensemble.RandomForestClassifier
    for more information and other possible parameters

    Args:
      random_state - (int) random seed
      n_jobs - (int or None) the number of jobs to run in parallel
                for both fit and predict. None means 1, -1 means
                using all processors
      n_estimators - (int) number of trees in the forest
      class_weights - (string) set "balanced" if class weights are to be used

    Returns:
      model - a sklearn random forest model
    """
    model = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs, n_estimators=n_estimators, class_weight=class_weight)
    return model

def make_logreg_model(random_state=None, solver='lbfgs', multi_class='multinomial'):
    """
    Defines a skearn logistic regression model. See ski-kit learn
    documentation of sklearn.linear_model.LogisticRegression for
    more information or other possible parameters

    Args:
      random_state - (int) random seed used to shuffle data
      solver - (str) {'newton-cg', 'lbfgs', 'linlinear', 'sag', 'saga'}
               for multiclass problems, only 'newton-cg', 'sag', 'saga',
               and 'lbfgs' handle multinomial loss. See docs for more info
      multiclass - (str) {'ovr', 'multinomial', 'auto'} for 'ovr', a
                   binary problem is fit for each label. For 'multinomial',
                   the minimized loss is the multinomial loss fit across
                   the entire probability distribution, even when binary.
                   See sci-kit learn docs for more information.

    Returns:
      model - a sklearn logistic regression model
    """
    model = LogisticRegression(random_state, solver, multi_class)
    return model

def make_1d_nn_model(num_classes, num_input_feats, units, reg_strength, input_bands, dropout):
    """ Defines a keras Sequential 1D NN model

    Args:
      num_classes - (int) number of classes to predict
      num_input_feats - (int) number of input features (timestamps)
      units - (int) corresponds to hidden layer features
      reg_stength - (float) constant for regularization strength for model weights
      input_bands - (int) number of input channels
      dropout - (float) constant for percentage of connections to drop during training

    Returns:
      loads self.model as the defined model
    """
    reg = regularizers.l2(reg_strength)

    model = Sequential()

    model.add(Flatten())
    model.add(Dense(units=units, kernel_regularizer=reg, 
              bias_regularizer=reg, input_shape=(num_input_feats, input_bands)))
    model.add(Activation('relu'))
    model.add(Dropout(rate=dropout))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax', 
              kernel_regularizer=reg, bias_regularizer=reg))

    return model

def make_1d_2layer_nn_model(num_classes, num_input_feats, units, reg_strength, input_bands, dropout):
    """ Defines a keras Sequential 1D NN model

    Args:
      num_classes - (int) number of classes to predict
      num_input_feats - (int) number of input features (timestamps)
      units - (int) corresponds to hidden layer features
      reg_stength - (float) constant for regularization strength for model weights
      input_bands - (int) number of input channels
      dropout - (float) constant for percentage of connections to drop during training

    Returns:
      loads self.model as the defined model
    """
    reg = regularizers.l2(reg_strength)

    model = Sequential()

    model.add(Flatten())
    model.add(Dense(units=units, kernel_regularizer=reg, 
              bias_regularizer=reg, input_shape=(num_input_feats, input_bands)))
    model.add(Activation('relu'))
    model.add(Dropout(rate=dropout))
    model.add(BatchNormalization())
    model.add(Dense(units=units, kernel_regularizer=reg,
              bias_regularizer=reg))
    model.add(Activation('relu'))
    model.add(Dropout(rate=dropout))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax', 
              kernel_regularizer=reg, bias_regularizer=reg))

    return model

def make_1d_cnn_model(num_classes, num_input_feats, units, reg_strength, input_bands, dropout):
    """ Defines a keras Sequential 1D CNN model

    Args:
      num_classes - (int) number of classes to predict
      num_input_feats - (int) number of input features (timestamps)
      units - (int) corresponds to hidden layer features
      reg_stength - (float) constant for regularization strength for model weights
      input_bands - (int) number of input channels
      dropout - (float) constant for percentage of connections to drop during training

    Returns:
      loads self.model as the defined model
    """
    reg = regularizers.l2(reg_strength)
    
    model = Sequential()

    model.add(Conv1D(units, kernel_size=3,
              strides=1, padding='same',
              kernel_regularizer=reg,
              bias_regularizer=reg,
              input_shape=(num_input_feats, input_bands)))
    model.add(Activation('relu'))
    model.add(Dropout(rate=dropout))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2, strides=2))

    model.add(Conv1D(units*2, kernel_size=3, padding='same',
              kernel_regularizer=reg, bias_regularizer=reg))
    model.add(Activation('relu'))
    model.add(Dropout(rate=dropout))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2, strides=2))

    model.add(Flatten())
    model.add(Dense(units*4, activation='relu', 
              kernel_regularizer=reg, bias_regularizer=reg))
    model.add(Dropout(rate=dropout))
    model.add(Dense(num_classes, activation='softmax', 
              kernel_regularizer=reg, bias_regularizer=reg))

    return model

