## Neural Networks: Zero to Hero (A Course by Andrej Karpathy)

A course on neural networks that starts all the way at the basics. The course is a series of YouTube videos where we code 
and train neural networks. 

---

**Lecture 1: The spelled-out intro to neural networks and backpropagation: building micrograd**

Backpropagation and training of neural networks. A base Value class is created containing basic operations like addition, multiplication and so on.
The Value class is then used in the Neuron class, the Layer and the MLP( Multi Layer Perceptron ) class. Everything is done manually the calculus for the backward pass, the gradients for each operation. Assumes basic knowledge of Python and a vague recollection of calculus from high school.

- [YouTube video lecture](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [Jupyter notebook files](/micrograd/Value.ipynb)

---

**Lecture 2: The spelled-out intro to language modeling: building makemore**

Implemented a bigram character-level language model. In this lesson, the focus is on (1) introducing torch.Tensor and its subtleties and use in efficiently
evaluating neural networks and (2) the overall framework of language modeling that includes model training, sampling, and the evaluation of a loss 
(e.g. the negative log likelihood for classification).

- [YouTube video lecture](https://www.youtube.com/watch?v=PaCmpygFfXo)
- [Jupyter notebook files](/makemore/makemore_part1_bigrams.ipynb)

---

**Lecture 3:  Building makemore Part 2: MLP**

Implemented a multilayer perceptron (MLP) character-level language model. In this lecture we also introduce many basics of machine learning (e.g. model training, learning rate tuning, hyperparameters, evaluation, train/dev/test splits, under/overfitting, etc.).
- [YouTube video lecture](https://www.youtube.com/watch?v=TCH_1BHY58I)
- [Jupyter notebook files](/makemore/makemore_part2_mlp.ipynb)

---

**Lecture 4: Building makemore Part 3: Activations & Gradients, BatchNorm**

Diving into some of the internals of MLPs with multiple layers and going over the statistics of the forward pass activations, backward pass gradients, and some of the pitfalls when they are improperly scaled. Using visualization tools to understand the health of the deep neural network.
Introducing Batch Normalization which makes the training much easier.
- [YouTube video lecture](https://www.youtube.com/watch?v=P6sfmUTpUmc)
- [Jupyter notebook file Part A](/makemore/makemore_part3_Activations_Gradients_BatchNorm.ipynb) 
- [Jupyter notebook file Part B](/makemore/makemore_part3_Activations_Gradients_BatchNorm_with_classes.ipynb)



---

**Building makemore Part 5: Building a WaveNet**

The 2-layer MLP from the previous video will be modified, we make it deeper with a tree-like structure, arriving at a convolutional neural network architecture similar to the WaveNet (2016) from DeepMind.
- [YouTube video lecture](https://www.youtube.com/watch?v=t3YJ5hKiMQ0)
- [Jupyter notebook files](/makemore/makemore_part5.ipynb)

---

**NanoGpt Lecture**

Generatively Pretrained Transformer (GPT), following the paper "Attention is All You Need" and OpenAI's GPT-2 / GPT-3.
There are a lot of comments in the files so check them for additional info.

- [YouTube video lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [NanoGpt directory](/NanoGpt/)





