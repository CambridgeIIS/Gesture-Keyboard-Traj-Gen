## Simulating Realistic Human Motion Trajectories of Mid-Air Gesture Typing

The eventual success of many AR and VR intelligent interactive systems relies on the ability to collect user motion data at large scale.
Realistic simulation of human motion trajectories is a potential solution to this problem. 
Simulated user motion data can facilitate prototyping and speed up the design process.
There are also potential benefits in augmenting training data for deep learning-based AR/VR applications to improve performance.
However, the generation of realistic motion data is nontrivial. 
In this paper, we examine the specific challenge of simulating index finger movement data to inform mid-air gesture keyboard design. The mid-air gesture keyboard is deployed on an optical see-through display that allows the user to enter text by articulating word gesture patterns with their physical index finger in the vicinity of a visualized keyboard layout.
We propose and compare four differen approaches to simulating this type of motion data, including a Jerk-Minimization model, a Recurrent Neural Network (RNN)-based generative model, and a Generative Adversarial Network (GAN)-based model with two modes: style transfer and data alteration.
We also introduce a procedure for validating the quality of the generated trajectories in terms of realism and diversity.
The GAN-based model shows significant potential for generating synthetic motion trajectories to facilitate design and deep learning for advanced gesture keyboards deployed in AR and VR. 

The implementations in this repository can enable readers better replicate our experiments and use the models as a rapid synthetic tools. 

<p align = "center">
<img src= "https://github.com/shawnshenjx/Gesture-Keyboard-Traj-Gen/blob/main/img/real_trace.jpg">
</p> 

## Getting started 

### 1. Jerk-Minimization

<p align = "center">
<img src= "https://github.com/shawnshenjx/Gesture-Keyboard-Traj-Gen/blob/main/img/jm_trace.jpg">
</p> 
*image_caption*

Implementation of the Jerk-Minimization Model. Code is built based on <a href="https://github.com/icsl-Jeon/traj_gen">traj_gen : a continuous trajectory generation with simple API </a>

Two examples are listed, to generate synthezied trajectories for demo:
- **optim_example** 
```
  $ python optim_example.py
```  
  
- **poly_example** 
```
  $ python poly_example.py
```

### 2. GAN-Based
<p align = "center">
<img src= "https://github.com/shawnshenjx/Gesture-Keyboard-Traj-Gen/blob/main/img/transfer_gan_data.jpg">
</p> 

<p align = "center">
<img src= "https://github.com/shawnshenjx/Gesture-Keyboard-Traj-Gen/blob/main/img/imitation_gan_data.jpg">
</p> 

Tensorflow 2 implementation of the Imaginative GAN with two modes - one is GAN-Transfer and the other is GAN-Imitation.
#### Usage
- **Setting dependencies** 
```
  $ pip install -r requrements.txt
```

- **Set Parameters**
```
  $ nano de_gan.gin
```

- **Train the model**
```
  $ python cycle_main.py
```
  
- **Generate synthezied trajectories**
 ```
  $ python inference.py
``` 

### 3. RNN-Based TF1
<p align = "center">
<img src= "https://github.com/shawnshenjx/Gesture-Keyboard-Traj-Gen/blob/main/img/rnn_trace.jpg">
</p> 
Tensorflow 1 implementation of the model in the paper <a href="https://arxiv.org/abs/1308.0850">Generating Sequences with Recurrent Neural Networks</a> by Alex Graves.

#### Usage
- **Setting dependencies** 
```
  $ pip install -r requrements.txt
    

- **Train the model**
 ```
  $ python run.py --train
```
  
- **Generate synthezied trajectories**
 ```
  $ python Os_run_auto.py
  

### 4. RNN-Based TF2



Tensorflow 2 implementation of the model in the paper <a href="https://arxiv.org/abs/1308.0850">Generating Sequences with Recurrent Neural Networks</a> by Alex Graves. Code is built based on <a href="https://github.com/sjvasquez/handwriting-synthesis">handwriting-synthesis</a>

#### Usage
- **Setting dependencies** 
```
  $ pip install -r requrements.txt
``` 

- **Train the model**
 ```
  $ python rnn.py
```
  
- **Generate synthezied trajectories**
 ```
  $ python drawing.py
```

