# Basic Machine Learning Question
I collect these questions from Machine Interviews Question book with the author is Huyen Chip, one of my favorite authors, who has some famous books for the youth in Vietnam.

By the way, this repository will be my working notebook. I will put somethings I learnt from School here, as well as some stuffs interested me.

# Questions and Answers
There are a lot of questions which are represented in the book, but I will list questions that I have no idea about them before.

- Explain weakly supervised, unsupervised, supervised, semi-supervised, and active learning.
    - Weakly supervised learning uses partially labeled or noisy data to train a model.
    - Active learning has the algorithm which is able to choose which data it wants (which is most informative) instead of being given a fixed set of data for training. This algorithm minimize the amount of labeled data required to achieve a desired level of accuracy.
 
- Empirical risk minimization.
    - What’s the risk in empirical risk minimization?
    - Why is it empirical?
    - How do we minimize that risk?
- Occam's razor states that when the simple explanation and complex explanation both work equally well, the simple explanation is usually correct. How do we apply this principle in ML?
- What are the conditions that allowed deep learning to gain popularity in the last decade?
- If we have a wide NN and a deep NN with the same number of parameters, which one is more expressive and why?
- The Universal Approximation Theorem states that a neural network with 1 hidden layer can approximate any continuous function for inputs within a specific range. Then why can’t a simple neural network reach an arbitrarily small positive error?
- What are saddle points and local minima? Which are thought to cause more problems for training large NNs?

## Hyperparameters.
- What are the differences between parameters and hyperparameters?
- Why is hyperparameter tuning important?
- Explain algorithm for tuning hyperparameters.
## Classification vs. regression.
- What makes a classification problem different from a regression problem?
- Can a classification problem be turned into a regression problem and vice versa?
## Parametric vs. non-parametric methods.
- What’s the difference between parametric methods and non-parametric methods? Give an example of each method.
- When should we use one and when should we use the other?
- Why does ensembling independently trained models generally improve performance?
- Why does L1 regularization tend to lead to sparsity while L2 regularization pushes weights closer to 0?
- Why does an ML model’s performance degrade in production?
- What problems might we run into when deploying large machine learning models?
## Realistic Problem
**Your model performs really well on the test set but poorly in production.**
- What are your hypotheses about the causes?
- How do you validate whether your hypotheses are correct?
- Imagine your hypotheses about the causes are correct. What would you do to address them?
