# My Mech Interp Journey

This is where I keep my notes and small experiments from learning mechanistic interpretability.

## Questions I want to answer

- What does the inside of an LLM actually compute, and can we read it?
- What tools do interpretability researchers use, and how mature are they?
- What are the major open problems worth caring about?

## What is inside

- `notes/`: short summaries of general mech-interp papers and blog posts.
- `01-computation-in-superposition/`: a learning project on superposition and computation in superposition.
- `02-memorisation-and-logitlens/`: two small experiments on where memorisation lives in a tiny transformer, and where factual answers emerge across layers in GPT-2 small.

More folders will land here over time.

## Concepts I've covered

The general mech-interp foundations.

- **Features and circuits.** Directions in activation space as features, connected by circuits across layers.
- **Polysemanticity.** Single neurons firing for many unrelated concepts.
- **Superposition.** How networks store more features than dimensions.
- **Toy-models methodology.** Tiny networks on controlled tasks to study one phenomenon at a time.
- **Probing.** Linear classifiers on activations to test whether a concept is linearly represented.
- **Sparse Autoencoders (SAEs).** What they are, what they claim, and the debate about whether they really recover the model's features.
- **TransformerLens basics.** Hooking activations and cached forward passes.

## Reading list

### Onboarding resources

- [Neel Nanda, "Getting Started in Mechanistic Interpretability"](https://www.neelnanda.io/mechanistic-interpretability/getting-started-old). A practical roadmap into the field.
- [ARENA](https://arena-chapter0-fundamentals.streamlit.app/). Hands-on curriculum with chapters on ML fundamentals and transformer interpretability.
- [TransformerLens, Getting Started](https://transformerlensorg.github.io/TransformerLens/content/getting_started.html). The library's onboarding docs.

### Foundational reads

1. ✅ [Chris Olah et al., "Zoom In: An Introduction to Circuits"](https://distill.pub/2020/circuits/zoom-in/) (Distill, 2020). The features-and-circuits framing.
2. [Elhage et al., "A Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html) (Anthropic, 2021). How transformer internals compose.
3. ✅ [Elhage et al., "Toy Models of Superposition"](https://transformer-circuits.pub/2022/toy_model/index.html) (Anthropic, 2022). Why and how networks pack more features than dimensions.
4. [Bricken et al., "Towards Monosemanticity: Decomposing Language Models with Dictionary Learning"](https://transformer-circuits.pub/2023/monosemantic-features/index.html) (Anthropic, 2023). The Sparse Autoencoder approach to finding features.
