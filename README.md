# Speech Emotion Recognition using Adversarial auto-encoders

For low-level acoustic features, Authors extract a set of 1582 features using the openSMILE toolkit. The set consists of an assembly of spectral prosody and energy based features. Authors use five folder cross validation scheme, but I use Cross validation scheme which is applied for speaker-independent manner. 

## Datasets
* Interactive Emotional Dyadic Motion Capture (IEMOCAP) database is required to run this code.
* It used only improvised data for context-independent situation evaluation.

## Dependencies
* openSMILE for low-level acoustic features extraction
* Tensorflow for Adversarial Auto-encoders
* scikit-learn for acurracy evaluation

## References
* [1] S. Sahu, R. Gupta, G. Sivaraman, W. AbdAlmageed, and C. Espy-Wilson, "Adversarial Auto-encoders for Speech Based Emotion Recogntion," in Proc. Interspeech, 2017.

## TODO
- [ ] parameter tunning



