# Intelligent Earth — Introduction to Machine Learning

**University of Oxford**

**Practical sessions · Michaelmas Term 2026**

Welcome to the practical component of **Introduction to Machine Learning**. These four hands-on sessions connect the ideas introduced in the lectures to small, runnable experiments in Python. You will work with linear regression, automatic differentiation, neural networks built from NumPy, and neural networks implemented in PyTorch.

The notebooks are designed for different levels of programming experience. You can run and investigate the supplied examples, complete selected coding tasks, and discuss questions or unexpected results with the demonstrators. **The aim is to understand the methods, not to finish every cell as quickly as possible.**

## Practical sessions at a glance

Each session is **two hours in the afternoon**.

| Session | Date | Topic | Main question |
| --- | --- | --- | --- |
| [Day 1](Day1/) | Monday, 2 November 2026 | Linear Regression | How do model assumptions and complexity affect fitting and generalisation? |
| [Day 2](Day2/) | Tuesday, 3 November 2026 | Autodiff and Neural Networks | How do forward computations, losses and gradients produce a learning update? |
| [Day 3](Day3/) | Thursday, 5 November 2026 | Neural Networks from Scratch | What calculations sit behind a neural-network layer and backpropagation? |
| [Day 4](Day4/) | Friday, 6 November 2026 | Neural Networks with PyTorch | How do we train and evaluate models while comparing architectures and regularisation? |

The sessions form a progression:

**Model fitting → gradients and optimisation → implementing network operations → training with a deep-learning framework.**

## How the notebooks are organised

Every practical separates activities into three types:

| Label | What to do |
| --- | --- |
| **CORE EXERCISE 1/N** | A priority activity for the two-hour session. Follow the accompanying **Task N** instruction and fill in the indicated code where required. |
| **PROVIDED EXAMPLE** | Working code or an explained example. Run it, inspect the result and consider the accompanying question. You do not need to rewrite the implementation. |
| **OPTIONAL EXERCISE** | Additional practice or a longer experiment. Attempt it if there is time, or continue after the session. |

A **Task N** label identifies the specific action to take and, where applicable, the corresponding code location. A `TODO` or `raise NotImplementedError` indicates an unfinished exercise; follow the instructions for that task before attempting dependent cells. In some optional notebooks, later cells cannot run until earlier functions have been implemented.

**Expectations:** Focus first on the core learning ideas and activities. It is fine to ask questions, compare observations and use provided code as a starting point. The longer optional sections are included for further exploration; completing every optional experiment during class is **not** expected.

## Getting started

1. Read [Setup Instructions](Setup_Instructions.md) before the first session. You may work in **Google Colab** or a **local Python environment**.
2. For a local environment, use the supplied [`requirements.txt`](requirements.txt). You do not need a pre-packaged environment.
3. Keep any supporting files in the same folder as their notebook. This is particularly important for Day 3 and Day 4.
4. Open the appropriate Day folder, select the notebook and run the initial setup or example cells in order.
5. Work through the **Core Exercises** first. Use the expected outcomes below to check what you have learned.

For an existing local Python installation, the essential setup commands are:

```bash
python -m venv .venv
# Activate .venv using the command for your operating system in Setup_Instructions.md
python -m pip install -r requirements.txt
```

See [Setup Instructions](Setup_Instructions.md) for environment activation, notebook kernels, Colab guidance, a quick import check and troubleshooting.

## Day 1 — Linear Regression

**Topics:** maximum likelihood estimation (MLE), prediction, bias/intercept, polynomial features, RMSE, model complexity and generalisation. Maximum a posteriori (MAP) estimation and Bayesian linear regression are included for further exploration.

**Core tasks (4)**

1. **Task 1 — Prediction:** use the fitted parameters to compute model predictions.
2. **Task 2 — Bias:** fit a model that includes an intercept and interpret its effect.
3. **Task 3 — Polynomial features:** construct the feature representation for a polynomial regression model.
4. **Task 4 — RMSE:** calculate prediction error and interpret the result.

**Expectation:** Explain what the intercept changes, why polynomial regression can model a curved relationship while remaining linear in its parameters, and why a low training error does not guarantee good performance on new data.

**Today's result:** You should be able to run a regression experiment, inspect the effect of changing model flexibility, and explain the difference between fitting the training observations and generalising to held-out observations.

**Optional:** Additional model-fit plots, MAP estimation and Bayesian linear regression, including uncertainty-aware predictions.

[Open Day 1 materials](Day1/)

## Day 2 — Automatic Differentiation and Neural Networks

**Topics:** PyTorch tensors, computation graphs, automatic differentiation, gradient accumulation, neural-network layers, MNIST classification and cross-entropy loss.

**Core tasks (3)**

1. **Task 1 — Gradients:** use automatic differentiation to compute and check a simple derivative.
2. **Task 2 — MNIST images:** visualise examples from the dataset and their labels.
3. **Task 3 — Classification loss:** compute cross-entropy from model logits and class labels.

**Expectation:** Read common tensor shapes, describe the role of `requires_grad=True` and `.backward()`, and follow the sequence **forward pass → loss → gradients → parameter update**. Understand that `cross_entropy` takes raw logits rather than probabilities produced by an additional softmax call.

**Today's result:** You should be able to inspect a small PyTorch computation, calculate a gradient, and connect a classification prediction to its training loss.

**Optional:** Full training/evaluation functions, prediction visualisation and a separate **Variational Autoencoder (VAE)** notebook. The VAE extension retains its six implementation tasks covering reparameterisation, loss, training, reconstructions, interpolation and sampling. Its tasks need to be completed before the full model can run.

[Open Day 2 materials](Day2/)

## Day 3 — Building a Neural Network with NumPy

**Topics:** parameter initialisation, matrix dimensions, forward propagation, activation functions, cost, backpropagation and gradient-descent updates.

**Core tasks (4)**

1. **Task 1 — Initialise parameters:** define the weights and biases with the expected shapes.
2. **Task 2 — Linear forward:** calculate `Z = W @ A + b` and check dimensions.
3. **Task 3 — Linear backward:** compute gradients with respect to activations, weights and biases.
4. **Task 4 — Update parameters:** apply gradient descent to the weights and biases.

**Expectation:** Identify the shapes of the objects in a network layer, explain why cached quantities are useful for backpropagation, and interpret how gradients update parameters.

**Today's result:** You should understand and implement the main computations behind a neural-network layer. The notebook develops reusable network **helper functions**; it does not require a complete end-to-end training experiment on an external dataset.

**Optional:** Independently implement the additional multilayer helper functions that are supplied as working examples along the main route.

**Important:** Keep `dnn_utils_v2.py`, `testCases_v4a.py` and the `images/` folder alongside the Day 3 notebook. Downloading only the notebook file may break imports and figures.

[Open Day 3 materials](Day3/)

## Day 4 — Neural Networks with PyTorch

**Topics:** CIFAR-10 image classification, data loading, multilayer perceptrons (MLPs), activation-function comparisons, convolutional neural networks (CNNs), regularisation, early stopping and checkpointing.

**Core tasks (2)**

1. **Task 1 — DataLoader:** construct the evaluation data loader without shuffling.
2. **Task 2 — MLP architecture:** define fully connected layers mapping flattened RGB images to 10 output logits.

**Expectation:** Relate the NumPy operations from Day 3 to PyTorch modules, tensors, losses and optimisers. Follow a forward/loss/update step, interpret model outputs, and discuss how training and evaluation curves reveal underfitting or overfitting.

**Today's result:** You should be able to set up an image-classification model in PyTorch, inspect its input and output shapes, and interpret results from a short training experiment. You do **not** need to run every long experiment to meet the core goals.

**Optional:** Compare ReLU, Sigmoid and Tanh; implement and train a CNN; experiment with regularisation and weight decay; and explore early stopping and model checkpointing.

### CIFAR-10 dataset

The notebook uses the [course-provided CIFAR-10 archive on Google Drive](https://drive.google.com/file/d/1yUOoLlV7yFkKx-gAYJTFbw6YPbIdumjq/view?usp=sharing), with [`prepare_cifar10_from_drive.py`](Day4/prepare_cifar10_from_drive.py) handling the local download and extraction when needed. Keep that script next to the notebook. The archive must contain the standard CIFAR-10 Python-format files; see [Setup Instructions](Setup_Instructions.md) for the expected format and troubleshooting.

**Allow time for the first download.** The long CNN experiments can take substantial time on a laptop CPU; for a short optional trial, reduce the epochs to 3–5 or consider a GPU runtime. The core activities do not require hours of training.

[Open Day 4 materials](Day4/)

## Working through a practical

A useful approach is to **predict → run or implement → observe → explain**. For example, before changing a polynomial degree or activation function, predict what might happen; after running the code, compare the result with your prediction and ask what explains any difference.

If an exercise takes longer than expected, move on with the provided examples and return to it later. At the end of each session, use the **Expectation** and **Today's result** sections above to check the key concepts you can explain—not just the number of cells you have executed.

For setup problems, start with [Setup Instructions](Setup_Instructions.md). During the practicals, ask a demonstrator if a code error, tensor shape or experimental result is unclear.
