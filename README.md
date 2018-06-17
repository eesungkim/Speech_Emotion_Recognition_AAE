# Speech Emotion Recognition using Adversarial auto-encoders

For low-level acoustic features, Authors extract a set of 1582 features using the openSMILE toolkit. The set consists of an assembly of spectral prosody and energy based features. Authors use five folder cross validation scheme, but this implementation is used one leave speaker cross validation scheme for speaker-independent manner. 

## Datasets
* Interactive Emotional Dyadic Motion Capture (IEMOCAP) database is required to run this code.

## Dependencies
* openSMILE for low-level acoustic features extraction
* Tensorflow for Adversarial Auto-encoders
* scikit-learn for classification and performance evaluation
