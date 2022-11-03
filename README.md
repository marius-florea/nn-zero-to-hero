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

Implemented a bigram character-level language model. In this video, the focus is on (1) introducing torch.Tensor and its subtleties and use in efficiently
evaluating neural networks and (2) the overall framework of language modeling that includes model training, sampling, and the evaluation of a loss 
(e.g. the negative log likelihood for classification).

- [YouTube video lecture](https://www.youtube.com/watch?v=PaCmpygFfXo)
- [Jupyter notebook files](/makemore/makemore_part1_bigrams.ipynb)

---

**Lecture 3:  Building makemore Part 2: MLP**
We implement a multilayer perceptron (MLP) character-level language model. In this video we also introduce many basics of machine learning (e.g. model training, learning rate tuning, hyperparameters, evaluation, train/dev/test splits, under/overfitting, etc.).
- [YouTube video lecture](https://www.youtube.com/watch?v=TCH_1BHY58I)
- [Jupyter notebook files](/makemore/makemore_part2_mlp.ipynb)
