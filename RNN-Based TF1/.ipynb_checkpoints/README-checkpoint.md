# Gesture Keyboard Fingertip Trjaectory Synthesis
Tensorflow 1 implementation of the model in the paper <a href="https://arxiv.org/abs/1308.0850">Generating Sequences with Recurrent Neural Networks</a> by Alex Graves.
## Usage
install the required packages 
```python
pip install -r requrements.txt
```
to train the model:
```python
python run.py --train
```
to generate synthezied trajectories:
```python
python Os_run_auto.py
```
by linking the phrase txt file you want to generate phrases on in the python file