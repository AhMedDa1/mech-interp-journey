# 01. Computation in Superposition

A learning project on superposition and computation in superposition.

## Questions I want to answer

- What is the difference between *representation* superposition and *computation* in superposition?
- In what regimes does computation in superposition actually appear in toy models?
- Why does the "Compressed Computation" setup not count as computation in superposition, and what would?
- What would a clean toy model of computation in superposition look like, one that mimics how real LLM concept representations seem to work?
- Is there any direct evidence of computation in superposition in real small LLMs?

## What is in here

- `notes/`: short summaries of CiS-specific papers and posts. General mech-interp summaries live in the top-level `notes/`.
- Experiment notebooks and scripts will land here as I run them.

Things I would like to try:

1. Replicate the basic representation-superposition result from Toy Models of Superposition. Small network learning to compress N > d features. See if I can reproduce the standard plots.
2. Walk through the "Compressed Computation" post carefully and reproduce its setup. Build my own intuition for why the author says it is not computation in superposition.
3. Try a small toy task that does look like computation in superposition, following the Hanni et al. or "Ping pong" formulation.

## Concepts I want to come away with

CiS-specific ideas I should be able to explain after this topic:

- **Representation vs computation superposition.** What it means to *use* a superposed feature in a non-linear op, not just store it.
- **What disqualifies a setup from being CiS.** The criteria from the Compressed Computation critique.
- **Where CiS appears in toy models.** The regimes (sparsity, geometry, task design) that lead to actual CiS, as in Hanni et al. and the Ping pong setup.

## Reading list

1. [Compressed Computation is (probably) not Computation in Superposition](https://www.lesswrong.com/posts/ZxFchCFJFcgysYsT9/compressed-computation-is-probably-not-computation-in) (LessWrong). A good first read for the topic. Even just the introduction is enough to orient.
2. [Toward A Mathematical Framework for Computation in Superposition](https://www.lesswrong.com/posts/2roZtSr5TGmLjXMnT/toward-a-mathematical-framework-for-computation-in) (LessWrong). A readable summary of the Hanni et al. paper.
3. [Ping pong computation in superposition](https://www.lesswrong.com/posts/g9uMJkcWj8jQDjybb/ping-pong-computation-in-superposition) (LessWrong). Another worked example.
4. [Circuits in Superposition: Compressing many small neural networks into one](https://www.lesswrong.com/posts/roE7SHjFWEoMcGZKd/circuits-in-superposition-compressing-many-small-neural) (LessWrong). A different framing of the same idea.
5. [Hanni et al., Mathematical Models of Computation in Superposition](https://arxiv.org/abs/2408.05451) (arXiv 2408.05451). The full paper.
