# Gesture Keyboard Fingertip Trjaectory Synthesis
Tensorflow 2 implementation of the Imaginative GAN with two modes - one is GAN-Transfer and the other is GAN-Imitation
## Usage
install the required packages 
```python
pip install -r requrements.txt
```
to train the model:
```python
python cycle_main.py
```
parameters can be set in
```python
de_gan.gin
```
to generate synthezied trajectories for demo:
```python
python inference.py
```
by setting the text you want the model to gesture in the file

