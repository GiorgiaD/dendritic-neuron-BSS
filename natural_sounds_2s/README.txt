This folder contains the database composed of 72 recordings of isolated natural sounds. 

The database contained 8 recordings of human speech from the EUSTACE (the Edinburgh University Speech Timing Archive and Corpus of English) speech corpus, 
23 recordings of animal vocalizations from the Animal Sound Archive, 
29 recordings of music instruments by Philharmonia Orchestra, 
and 12 sounds produced by inanimate objects from the BBC Sound Effect corpus. 

The sounds were cut into 800ms extracts. 

Then the python-based library librosa was employed to extract spectrograms with 128 frequency filters spaced following the Mel scale 
and 10ms time-frames with 50\% overlap.
