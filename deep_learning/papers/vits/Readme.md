Components
- Posterior encoder
- Prior encoder
- Decoder
- Discriminator
- Stochastic duration predictor

Posterior encoder:
    Encodes the input audio waveform

Prior encoder:
    Text normalization -> Phoneme converter -> Text encoder (Attn, relative pos embed.)


Summarization 
0. Give the main contribution of the whitepaper in 3-4 short sentences. 
1. What exact problem does this paper aim to solve. Give a one-pager background on the problem and state of the art.
2. How does this paper build on the prior work and what improvements have they done.
2. Identify, list and explain the key terms or acronyms used in the paper. 
3. Draw (or describe) the high-level architecture or pipeline proposed in the paper. 
4. High level training data format (for example- audio-text pairs obtained from the internet, or non-parallel data)
5. Is there any official or third-party code for this paper? If so, where is it (GitHub, Colab, etc.)?

Architecture
0. Draw the architecture or use images from the paper to display the main architecture
1. Write down the step-by-step algorithm in plain English, as if explaining to a non-researcher. 
2. What is the input and output of the model or method? 
3. What are the core mathematical equations, and what do the variables represent ? 
4. What are the main high level components used in the architecture. List down the components along with their input/output. 
5. For each component list down - input/output, if the component is new or the paper reuses an existing one, if the paper makes any improvements to the component if it's already existing, if yes then what improvment (in 2-3 sentences), training methodology for the component or if pre-trained

Training
0. List down the datasets used for training.
1. List all parameters, and hyperparameters mentioned. 
2. What are the key training nuances (e.g., loss function, optimizer, num blocks, steps etc.)? 
3. What where the compute requirements for training. Convert and list down the training time in terms of A100 x8, H100 x8 GPU wall time.

Results
1. What are the main metrics used for quantifying results. List down all metrics along with their definitions.
2. State the baseline number for each metric along with the numbers that this paper produces. Always display this as a table.

Implementation 
1. Break down the main modules/classes that will be needed to implement the method/paper.
2. Keep each component in it's own class with separation of concerns.
3. Create pseudocode for each component and then map pseudocode to code.
4. For components that exist, try to reuse existing stuff as far as possible, eg. use huggingface libraries or pytorch implementations.
5. For components that exist, but have been modified by the paper, implement them with the required changes.
6. Write a config class to encode all config required by all components (based on the paper).
7. Create methods that load the required datasets from huggingface or via direct download and load (if needed).
8. Create methods that preprocess that data as required by the paper.
9. Write a training loop, that uses the created model architecture and trains it on the basis of hyperparameters and other training parameters specified in the paper.

