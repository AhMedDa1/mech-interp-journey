# Olah et al., Zoom In: An Introduction to Circuits

Distill, 2020. https://distill.pub/2020/circuits/zoom-in/

## Summary

A short Distill paper that lays out the case for treating neural networks as objects that can be reverse engineered, like a program. Most of the examples are from InceptionV1, a vision model. The paper does not prove anything formally; it sets out three speculative claims that the rest of mechanistic interpretability is built on.

## The three claims

### 1. Features

The basic unit of representation in a neural network is a feature, and a feature is a direction in activation space. It can be axis-aligned with a single neuron, but very often it is not. Olah argues we should think in terms of features instead of neurons.

Examples from the paper:

- Curve detectors in InceptionV1. Specific neurons fire for curves at specific orientations. The detectors appear at multiple layers, with deeper ones built from simpler ones.
- High-low frequency detectors. Neurons that respond to textures with mixed high and low spatial frequencies.
- Dog-head detectors further into the network. Higher level features built out of combinations of lower-level ones.

### 2. Circuits

Features in one layer connect to features in the next through the weights. A circuit is a small subgraph of features connected by their weights, and it computes something specific.

Examples from the paper:

- Curve circuits. Curve detectors at one orientation feed into curve detectors at neighbouring orientations in later layers. Olah walks through the connection weights to show this.
- Dog-head circuit. A dog-head feature is built from earlier features like fur textures, snout shapes, and eye detectors.

### 3. Universality

Similar features and similar circuits should appear in different networks trained on similar data. If true, mech interp results would generalise across models, not just describe one specific model.

The paper points to prior work showing curve-like detectors across multiple vision models as evidence, but Olah is honest that this is the most speculative of the three claims.

## What I am thinking

The three claims feel almost obvious in 2025, because the field absorbed them. In 2020 they were a real shift, since most people still treated networks as black boxes.

The universality claim is the most interesting because it implies mech interp findings could generalise. If different networks really do learn the same features and circuits when trained on similar data, then studying one model teaches you about a whole class of them. That would justify the whole research program.

The methodology in the paper is concrete. Pick a neuron, look at what it fires for, look at the weights into it, look at the weights out of it. Reverse engineering as a literal activity, not just a metaphor.

Polysemanticity is the major open problem the paper flags. Many neurons fire for several unrelated concepts, which breaks the clean "one neuron equals one feature" picture. The paper leaves it unsolved.

## What I have learned

- Features are directions in activation space, not necessarily neurons.
- Circuits are small subgraphs of features connected by weights.
- Universality is a hypothesis that mech interp findings generalise across models.
- Polysemanticity is the main open problem at the time of writing.