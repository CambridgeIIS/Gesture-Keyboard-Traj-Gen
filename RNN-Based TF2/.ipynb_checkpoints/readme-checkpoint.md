# Gesture Keyboard Fingertip Trjaectory Synthesis
Tensorflow 2 implementation of the model in the paper <a href="https://arxiv.org/abs/1308.0850">Generating Sequences with Recurrent Neural Networks</a> by Alex Graves. Code is built based on <a href="https://github.com/sjvasquez/handwriting-synthesis">handwriting-synthesis</a>
## Usage
install the required packages 
```python
pip install -r requrements.txt
```
to train the model:
```python
python rnn.py
```
to generate synthezied trajectories for demo:
```python
python drawing.py
```
by setting the text you want the model to gesture in the file