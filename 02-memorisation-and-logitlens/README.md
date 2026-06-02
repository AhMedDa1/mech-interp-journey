# 02. Memorisation and LogitLens

Two small experiments to get a feel for where memorisation and factual recall live inside small transformer-style models. The first trains a tiny one-layer transformer on a key-to-value memorisation task and asks whether the MLP is actually doing the work we usually credit it with. The second runs LogitLens on GPT-2 small over a list of factual prompts to see at which layer the answer emerges.

## Questions I want to answer

- Can a tiny one-layer transformer memorise many more key-value pairs than it has MLP neurons?
- Where in the model does that memorisation actually live: the MLP, the attention layer, or the embed and unembed?
- If I upgrade the task to require non-linear computation across multiple keys (XOR over a pair of values), does the MLP become essential?
- On GPT-2 small, at which layers does the correct token for a factual prompt emerge in the residual stream?
- What are the obvious confounders in the GPT-2 measurement, and how would I redo it more carefully?

## What is in here

- `CiS.ipynb`. The working notebook with both experiments, plots, and the small follow-up experiments (MLP ablation, XOR red-team, full LogitLens sweep across 1280 facts).
- `report.pdf`. A short writeup of what I found, with the figures pulled from the notebook.

Related: the `01-computation-in-superposition/` folder has my reading project on the topic, which sets up the vocabulary the first experiment in this folder is testing.

## Concepts I've come away with

- A tiny transformer with `d_model=32` and `d_mlp=128` can memorise 4096 key-to-value pairs, a 32x ratio of pairs to MLP neurons. So the answer to the first question is yes.
- Memorisation is not actually living in the MLP. Ablating the MLP after training drops accuracy from 1.000 to 0.826, against a random baseline of 0.001. Most of the work sits in `embed -> attention -> unembed` with the MLP adding roughly 17 points of refinement.
- A natural "make it CiS" upgrade does not work the way I expected. Switching to multi-token input with an XOR target and ablating the MLP still leaves accuracy above 99% in most configs. The softmax inside attention is itself a non-linearity and can absorb XOR-like routing. To isolate the MLP cleanly you have to constrain the attention layer.
- On GPT-2 small, factual answers do not pop in at a single layer. The correct token's rank improves gradually across layers 6 to 10 out of 12, and the median first-top-1 layer over emerging facts is 9. Different categories peak at slightly different layers, but the overall picture is smooth, not localised.
- First-token-only accuracy badly underestimates what GPT-2 knows. Three categories scored 0% but had near-1.0 AUC across layers, which means the model knew the answer but its first token was something like a leading "the" that the metric did not credit.

## Reading list

Foundational reads I used to set up the vocabulary:

1. ✅ **Chris Olah et al., "Zoom In: An Introduction to Circuits"** (Distill, 2020). https://distill.pub/2020/circuits/zoom-in/
   The paper that set the modern mech interp vocabulary: features as directions in activation space, circuits as small subgraphs of features connected by weights, universality as a hypothesis that the same features and circuits show up across networks. Flagged polysemanticity as the central open problem.

2. ✅ **Elhage et al., "Toy Models of Superposition"** (Anthropic, 2022). https://transformer-circuits.pub/2022/toy_model/index.html
   Demonstrates that a network can store more features than it has dimensions when the features are sparse, by packing them as non-orthogonal directions. Polysemanticity falls out of this naturally and the paper maps the regime where it appears.

Computation in superposition:

3. ✅ **"Compressed Computation is (probably) not Computation in Superposition"** (LessWrong). https://www.lesswrong.com/posts/ZxFchCFJFcgysYsT9/compressed-computation-is-probably-not-computation-in
   Argues that the canonical "compressed computation" toy demonstration is solved by compressed storage plus a linear readout, not by per-feature non-linear computation. Introduces a noise-sensitivity protocol that catches trivial-solution artefacts. This is the methodological pattern I tried to copy when I red-teamed my own setup.

4. **"Toward A Mathematical Framework for Computation in Superposition"** (LessWrong). https://www.lesswrong.com/posts/2roZtSr5TGmLjXMnT/toward-a-mathematical-framework-for-computation-in
   Readable summary of the Hänni framework: a mathematical proposal for how a single MLP could perform non-linear computations across many features stored in superposition.

For the GPT-2 part I used LogitLens, which was originally introduced by nostalgebraist on LessWrong. The technique just projects the intermediate residual stream through the final LayerNorm and unembed at each layer, treating each layer's residual as if it were a final-layer output.
