# ImageCaptionGenerator

## Important Information

a) Shree Murthy, Dylan Inafuku, Rahul Sura

b) shmurthy@chapman.edu, dinafuku@chapman.edu, sura@chapman.edu

c) CPSC 393-02

d) Final Project - Image Caption Generator

## Final Deliverables (Corresponding Links are in the following section)
- Report
- glove_unzip.py (used to unzip the glove_embeddings)
- feature_extraction.py
- model.py
- README.md (This file)


## Important Links 
- [Report](report/Report.pdf)
- [Proposal](proposal/Proposal.pdf)
- [Code](src/)
    - [glove_unzip.py](src/glove_unzip.py)
    - [feature_extraction.py](src/feature_extraction.py)
    - [model.py](src/model.py)
- [Flicker8k Kaggle Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- [Flicker Source Documentation](https://hockenmaier.cs.illinois.edu/8k-pictures.html)

## Running project

Download the Flicker8k Kaggle Dataset and place it in the src/ directory:
- Dataset is linked at bottom of this README

If you need to run the feature extraction model then run:

```
python3 src/feature_extraction.py
```


Download this GloVe embedding file and place it in the src/:

- [GloVe Embedding File](http://nlp.stanford.edu/data/glove.6B.zip)
- After downloading the file, unzip it using the glove_unzip.py file in the src/ directory by running:
```
python3 src/glove_unzip.py
```

To run this project it is very simple:

```
python3 src/model.py
```

Follow the prompts as they arise. If you are planning on using images that aren't in the dataset please upload them to the repositories src/ directory and then pass in the name of the png file.
Ex: If I have a file named play-in-park.png then that file must exist in my src/ folder and when I'm prompted to provide an image I will pass in the name `play-in-park.png`

If you would like a demo here is a link:
[Video](https://clipchamp.com/watch/t6sN1LIRw4v)


## Replicating Findings

Refer to the proposal pdf document, that is linked at the bottom of this README, to understand overarching architecture and approach. 

## Important Prequisities

Download the following python libraries:
NOTE: If pip doesn't work try pip3
```
pip install tensorflow

**Keras is installed as part of this pip install
``` 

```
pip install pickle
```

## Contributions

Shree Murthy:
- Created the .py files for the project
- Created and Coded Encoder/Decoder Model
- Created the feature that accepts any image to be inputted and captioned  
- Typed up the Proposal and Final Report with assistance from Dylan and Rahul

Rahul Sura:
- Helped find resources to build the encoder/decoder model 
- Helped Shree with the Proposal and Final Report
- Debugged the Code
- Wrote the Image Extraction Feature using InceptionV3

Dylan Inafuku:
- Helped Shree create the feature that accepts any image to be inputted and captioned
- Helped Shree with the Proposal and Final Report
- Created the README.md file and Final Presenation slidedeck 
- Found the GloVe embedding file(s) and helped implement it into the model


## Errors

- No compiler or other errors (everything ran/compiled)
