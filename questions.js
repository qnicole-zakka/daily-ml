window.QUESTION_BANK = [
  {
    "id": "q01",
    "type": "qa",
    "topic": "Basic deep learning",
    "title": "Logits or probabilities?",
    "prompt": "Your classifier produces raw logits of shape (batch, classes). Should you apply softmax before passing them to PyTorch cross_entropy? Explain why.",
    "answer": [
      "Pass raw logits directly to cross_entropy. It combines log-softmax with negative log-likelihood using a numerically stable calculation. Applying softmax first makes cross_entropy treat probabilities as logits, changing the loss and gradients. Use softmax separately when you want probabilities for inspection or prediction."
    ],
    "takeaway": "Pass raw logits to cross_entropy; use softmax separately when you need probabilities."
  },
  {
    "id": "q02",
    "type": "qa",
    "topic": "Basic deep learning",
    "title": "Diagnose a widening validation gap",
    "prompt": "Training loss keeps falling, but validation loss starts rising. What would you check first, and what are two reasonable responses?",
    "answer": [
      "This pattern suggests overfitting, but first check that training and validation use comparable loss calculations and validation runs in eval mode. Confirm the split matches the intended deployment setting and preprocessing is fitted only on training data. Then consider early stopping at the best validation checkpoint and stronger regularization, such as weight decay or suitable data augmentation. Choose changes using validation results; keep the test set for the final evaluation."
    ],
    "takeaway": "Validate the evaluation setup, then choose remedies using validation data."
  },
  {
    "id": "q03",
    "type": "qa",
    "topic": "Transformers",
    "title": "Why scale attention scores?",
    "prompt": "Scaled dot-product attention uses softmax(QK^T / sqrt(d_k)). Why divide by sqrt(d_k), and which dimension receives the softmax?",
    "answer": [
      "If query and key components are roughly independent with unit variance, their dot product has variance proportional to d_k. Dividing by sqrt(d_k) keeps score magnitudes more consistent as the key dimension grows and reduces softmax saturation. Apply softmax over the key positions for each query, so each query gets a distribution over the tokens it can attend to. Mask disallowed keys before softmax."
    ],
    "takeaway": "Scale by the square root of the head dimension; normalize over keys."
  },
  {
    "id": "q04",
    "type": "qa",
    "topic": "Transformers",
    "title": "Causal attention and next-token labels",
    "prompt": "For a decoder-only training sequence [BOS, A, B, EOS], write the input tokens and next-token labels. In an attention matrix where rows are queries and columns are keys, which positions must be masked?",
    "answer": [
      "Use inputs [BOS, A, B] and labels [A, B, EOS]. The output at each input position predicts the following token. Mask every key column j greater than the query row i: those keys are future tokens. The diagonal is allowed. Shifting the labels sets the prediction task; the causal mask prevents the model from seeing future inputs while solving it."
    ],
    "takeaway": "Predict the next token and mask future keys. The diagonal remains visible."
  },
  {
    "id": "q05",
    "type": "qa",
    "topic": "LLM fine-tuning",
    "title": "What does LoRA actually save?",
    "prompt": "Compare full fine-tuning and LoRA. Which parameters receive updates, and why can LoRA reduce training memory without eliminating the base model memory cost?",
    "answer": [
      "Full fine-tuning updates the selected original weights. LoRA freezes a base weight W and learns a low-rank update BA, often in selected attention or feed-forward projections. Far fewer parameters need gradients and optimizer state, so training memory can fall substantially. The base weights still have to be stored and used in the forward pass, and activations still cost memory. LoRA alone does not quantize the base model."
    ],
    "takeaway": "LoRA reduces trainable parameters and optimizer state; the base model and activations still consume memory."
  },
  {
    "id": "q06",
    "type": "qa",
    "topic": "LLM inference",
    "title": "Prefill, decode, and the KV cache",
    "prompt": "What happens during prompt prefill and autoregressive decode? What work does a KV cache avoid, and does it make attention cost independent of context length?",
    "answer": [
      "Prefill processes the prompt and builds keys and values for its tokens at each attention layer. Decode generates tokens one at a time, adding each new token to the cache. The cache avoids recomputing earlier keys and values on every step. A new query still attends to the growing history, so reading and scoring that history costs more as the context grows. Cache memory also grows with the number of stored tokens."
    ],
    "takeaway": "A KV cache reuses prior keys and values, but attention still reads the growing context."
  },
  {
    "id": "q07",
    "type": "coding",
    "topic": "Basic deep learning",
    "title": "Implement a stable classification loss",
    "prompt": "Implement classification_loss(logits, targets) for multiclass classification. logits has shape (batch, classes); targets contains integer class indices. Use the stable PyTorch loss directly without softmax first. The checks include large logits and a backward pass.",
    "answer": [
      "cross_entropy expects unnormalized logits and integer class indices. Its stable log-softmax calculation avoids taking the logarithm of rounded probabilities. The default reduction averages across examples. These inputs produce a loss of about 0.813262 and finite gradients."
    ],
    "starterCode": "import torch\nimport torch.nn.functional as F\n\ndef classification_loss(logits, targets):\n    # TODO: return the mean multiclass cross-entropy loss.\n    raise NotImplementedError\n\nlogits = torch.tensor([[1000., 999., -1000.],\n                       [-1000., 1000., 999.]], requires_grad=True)\ntargets = torch.tensor([0, 2], dtype=torch.long)\nloss = classification_loss(logits, targets)\nassert torch.isfinite(loss)\nloss.backward()\nassert torch.isfinite(logits.grad).all()\nprint(f\"loss = {loss.item():.6f}\")",
    "solutionCode": "import torch\nimport torch.nn.functional as F\n\ndef classification_loss(logits, targets):\n    return F.cross_entropy(logits, targets)\n\nlogits = torch.tensor([[1000., 999., -1000.],\n                       [-1000., 1000., 999.]], requires_grad=True)\ntargets = torch.tensor([0, 2], dtype=torch.long)\nloss = classification_loss(logits, targets)\nassert torch.isfinite(loss)\nassert torch.allclose(loss, torch.tensor(0.8132617), atol=1e-6)\nloss.backward()\nassert torch.isfinite(logits.grad).all()\nprint(f\"loss = {loss.item():.6f}\")",
    "takeaway": "Stable cross-entropy accepts logits and produces finite gradients even for large scores.",
    "checks": [
      "The printed mean loss is approximately 0.813262.",
      "Both the loss and every logit gradient must be finite."
    ]
  },
  {
    "id": "q08",
    "type": "coding",
    "topic": "Transformers",
    "title": "Build a causal attention head",
    "prompt": "Implement one attention head for a single sequence. q, k, and v have shape (sequence_length, head_dim). Scale q @ k.T, mask future key positions, and apply softmax over keys. Return (output, weights). Ignore padding and dropout. With zero queries and keys, each token should average only its own and earlier values.",
    "answer": [
      "Rows represent queries and columns represent keys. A True value above the diagonal marks a future key, so replace those scores with negative infinity before softmax. Softmax along the final dimension normalizes the allowed keys for each query. Uniform allowed scores give the expected running averages."
    ],
    "starterCode": "import math\nimport torch\n\ndef causal_attention(q, k, v):\n    # TODO: scaled scores, causal mask, softmax, weighted values.\n    raise NotImplementedError\n\nq = torch.zeros(3, 2)\nk = torch.zeros(3, 2)\nv = torch.tensor([[1., 1.], [2., 2.], [3., 3.]])\noutput, weights = causal_attention(q, k, v)\nexpected = torch.tensor([[1., 1.], [1.5, 1.5], [2., 2.]])\nassert torch.allclose(output, expected)\nassert torch.allclose(weights.sum(dim=-1), torch.ones(3))\nassert torch.equal(weights.triu(1), torch.zeros_like(weights))\nprint(output)",
    "solutionCode": "import math\nimport torch\n\ndef causal_attention(q, k, v):\n    scores = (q @ k.T) / math.sqrt(q.shape[-1])\n    future = torch.ones_like(scores, dtype=torch.bool).triu(1)\n    weights = torch.softmax(scores.masked_fill(future, -torch.inf), dim=-1)\n    return weights @ v, weights\n\nq = torch.zeros(3, 2)\nk = torch.zeros(3, 2)\nv = torch.tensor([[1., 1.], [2., 2.], [3., 3.]])\noutput, weights = causal_attention(q, k, v)\nexpected = torch.tensor([[1., 1.], [1.5, 1.5], [2., 2.]])\nassert torch.allclose(output, expected)\nassert torch.allclose(weights.sum(dim=-1), torch.ones(3))\nassert torch.equal(weights.triu(1), torch.zeros_like(weights))\nv_changed = v.clone()\nv_changed[-1] = 999.\nchanged_output, _ = causal_attention(q, k, v_changed)\nassert torch.allclose(changed_output[:2], output[:2])\nprint(output)",
    "takeaway": "Mask future scores before softmax so later tokens cannot affect earlier outputs.",
    "checks": [
      "The outputs are [1, 1], [1.5, 1.5], and [2, 2].",
      "Each attention row sums to one; every weight above the diagonal is zero.",
      "Changing the final value token must leave the first two outputs unchanged."
    ]
  },
  {
    "id": "q09",
    "type": "coding",
    "topic": "LLM fine-tuning",
    "title": "Train a tiny LoRA adapter",
    "prompt": "Complete a single CPU training step for a linear layer with a rank-one adapter. Freeze W, implement x @ W.T + (x @ A.T) @ B.T, and optimize only A and B. Initialize B to zero so the adapter initially leaves the base output unchanged. Verify that W stays fixed. Use adapter scale 1 and no bias.",
    "answer": [
      "Freeze W with requires_grad_(False), and give the optimizer only A and B. Two nested linear operations compute the rank-one update without constructing a dense BA matrix. With B initially zero, the initial prediction exactly matches the base layer. On the first backward pass the gradient of A is zero, while B can update; A begins receiving a nonzero gradient after B changes."
    ],
    "starterCode": "import torch\nimport torch.nn.functional as F\n\ntorch.manual_seed(0)\nW = torch.nn.Parameter(torch.randn(2, 3))\n# TODO: freeze W.\nA = torch.nn.Parameter(torch.randn(1, 3) * 0.1)\nB = torch.nn.Parameter(torch.zeros(2, 1))\n\ndef forward(x):\n    # TODO: base output plus the low-rank update.\n    raise NotImplementedError\n\n# TODO: create an SGD optimizer for A and B only, with lr=0.1.\noptimizer = None\nx = torch.tensor([[1., 2., 3.], [-1., 0., 1.]])\ntarget = torch.zeros(2, 2)\nW_before = W.detach().clone()\nassert torch.allclose(forward(x), F.linear(x, W))\noptimizer.zero_grad()\nloss = F.mse_loss(forward(x), target)\nloss.backward()\noptimizer.step()\nassert W.grad is None\nassert torch.equal(W, W_before)\nassert torch.count_nonzero(B).item() > 0\nprint(f\"loss before update = {loss.item():.6f}\")",
    "solutionCode": "import torch\nimport torch.nn.functional as F\n\ntorch.manual_seed(0)\nW = torch.nn.Parameter(torch.randn(2, 3))\nW.requires_grad_(False)\nA = torch.nn.Parameter(torch.randn(1, 3) * 0.1)\nB = torch.nn.Parameter(torch.zeros(2, 1))\n\ndef forward(x):\n    return F.linear(x, W) + F.linear(F.linear(x, A), B)\n\noptimizer = torch.optim.SGD([A, B], lr=0.1)\nx = torch.tensor([[1., 2., 3.], [-1., 0., 1.]])\ntarget = torch.zeros(2, 2)\nW_before = W.detach().clone()\nassert torch.allclose(forward(x), F.linear(x, W))\noptimizer.zero_grad()\nloss = F.mse_loss(forward(x), target)\nloss.backward()\nassert A.grad is not None and torch.count_nonzero(A.grad).item() == 0\nassert B.grad is not None and torch.count_nonzero(B.grad).item() > 0\noptimizer.step()\nassert W.grad is None\nassert torch.equal(W, W_before)\nassert torch.count_nonzero(B).item() > 0\nprint(f\"loss before update = {loss.item():.6f}\")",
    "takeaway": "Freeze the base weights and optimize only the adapters. Zero-initialized B makes the first A gradient zero.",
    "checks": [
      "The initial adapter contribution is zero, so predictions match the frozen base layer.",
      "W stays unchanged and has no gradient; B changes after the step.",
      "At initialization, A has a zero gradient because B is zero."
    ]
  },
  {
    "id": "q10",
    "type": "coding",
    "topic": "LLM inference",
    "title": "Sample with temperature and top-k",
    "prompt": "Implement sample_token for a finite, one-dimensional floating-point logits tensor on CPU. Keep only the top_k largest logits, apply a finite positive temperature, and sample one original token index. Reject nonpositive or nonfinite temperatures and top_k outside 1..vocabulary_size. Center the selected logits before dividing by temperature to avoid positive overflow. top_k is an integer.",
    "answer": [
      "topk returns selected values and their original token indices. Subtract the largest selected logit before dividing by temperature; this preserves the softmax distribution and prevents positive overflow. Softmax produces probabilities over the selected tokens. multinomial samples a position in that smaller list; map it back to the original vocabulary index. Lower temperature concentrates probability on larger logits; top_k=1 gives greedy selection."
    ],
    "starterCode": "import math\nimport torch\n\ndef sample_token(logits, temperature=1.0, top_k=2, generator=None):\n    # TODO: validate settings, select top-k, scale, softmax, sample.\n    raise NotImplementedError\n\nlogits = torch.tensor([1000., 999., -1000.])\ngenerator = torch.Generator().manual_seed(0)\nsamples = [sample_token(logits, 0.5, 2, generator) for _ in range(20)]\nassert set(samples) <= {0, 1}\nassert sample_token(logits, 1.0, 1, generator) == 0\nprint(samples)",
    "solutionCode": "import math\nimport torch\n\ndef sample_token(logits, temperature=1.0, top_k=2, generator=None):\n    if not math.isfinite(temperature) or temperature <= 0:\n        raise ValueError(\"temperature must be finite and positive\")\n    if not 1 <= top_k <= logits.numel():\n        raise ValueError(\"top_k must be between 1 and vocabulary_size\")\n    values, token_ids = torch.topk(logits, top_k)\n    scaled = (values - values.max()) / temperature\n    probabilities = torch.softmax(scaled, dim=-1)\n    position = torch.multinomial(probabilities, 1, generator=generator)\n    return token_ids[position].item()\n\nlogits = torch.tensor([1000., 999., -1000.])\ngenerator = torch.Generator().manual_seed(0)\nsamples = [sample_token(logits, 0.5, 2, generator) for _ in range(20)]\nassert set(samples) <= {0, 1}\nassert sample_token(logits, 1.0, 1, generator) == 0\nassert sample_token(torch.tensor([1e38, 0., -1e38]), 0.01, 2, generator) == 0\nfor temperature, top_k in [(0., 2), (float(\"nan\"), 2), (1., 0), (1., 4)]:\n    try:\n        sample_token(logits, temperature, top_k, generator)\n    except ValueError:\n        pass\n    else:\n        raise AssertionError(\"invalid sampling settings were accepted\")\nprint(samples)",
    "takeaway": "Sample within top-k, then map the sampled position back to its vocabulary index.",
    "checks": [
      "Samples come only from the two highest-logit token IDs.",
      "With top_k=1, the highest-logit token is always returned.",
      "Invalid temperatures or top-k bounds raise ValueError."
    ]
  },
  {
    "id": "q11",
    "type": "qa",
    "topic": "Basic deep learning",
    "title": "Trace a gradient through the chain rule",
    "prompt": "For the scalar loss L = (w × x)², let x = 3 and w = 2. Calculate dL/dw by introducing z = w × x. Why does backprop multiply local derivatives?",
    "answer": [
      "Here z = 6, dL/dz = 2z = 12, and dz/dw = x = 3, so dL/dw = 12 × 3 = 36. The chain rule combines how each operation changes its output with how later operations change the loss. Backprop applies this rule through the computation graph; when a value influences the loss along multiple branches, its gradient adds the contributions from those branches."
    ],
    "takeaway": "Multiply derivatives along each path, then add gradient contributions from different paths."
  },
  {
    "id": "q12",
    "type": "qa",
    "topic": "Basic deep learning",
    "title": "Why can a ReLU unit stop learning?",
    "prompt": "A unit uses ReLU(z) = max(0, z), and z is negative for every training example. What gradient passes through this activation, and how could a leaky ReLU help?",
    "answer": [
      "For negative z, ReLU has derivative zero, so the loss sends no gradient through that activation to the unit's incoming weights for those examples. If the unit remains negative across the data, it can become inactive. A leaky ReLU uses a small positive slope on the negative side, allowing a gradient to pass. Initialization and learning rate also influence whether units enter this state; changing the activation is not a guaranteed cure for every training problem."
    ],
    "takeaway": "A zero activation derivative blocks that gradient path; a leaky negative slope keeps it open."
  },
  {
    "id": "q13",
    "type": "qa",
    "topic": "Basic deep learning",
    "title": "Make gradient accumulation match a larger batch",
    "prompt": "You split one batch into four equal microbatches, each returning a mean loss. How should you scale the losses and schedule zero_grad and optimizer.step to approximate one large-batch update?",
    "answer": [
      "Clear gradients once before the group, divide each microbatch's mean loss by four, and call backward for each. Step the optimizer once after all four backward passes. This adds gradients with the same averaging as the combined batch, assuming independent example losses and unchanged parameters during accumulation. Unequal microbatches need weighting by example count; token-averaged losses need weighting by valid token count. Batch-dependent layers and different random dropout masks can prevent exact numerical equivalence."
    ],
    "takeaway": "Accumulate correctly weighted gradients before taking one optimizer step."
  },
  {
    "id": "q14",
    "type": "qa",
    "topic": "Basic deep learning",
    "title": "Separate evaluation mode from gradient recording",
    "prompt": "For a model containing dropout, what changes with model.eval()? Does it also disable autograd, and what happens if you only use torch.no_grad() while the model stays in training mode?",
    "answer": [
      "Evaluation mode disables dropout's random masking; training-mode dropout scales surviving activations by 1/(1 − p). model.eval() changes module behavior but does not disable autograd. Conversely, torch.no_grad() disables gradient recording without switching module behavior, so training-mode dropout remains random inside it. For ordinary evaluation, use evaluation mode together with no_grad or inference_mode. BatchNorm also changes its usual behavior between training and evaluation."
    ],
    "takeaway": "Module mode controls dropout behavior; gradient contexts control autograd recording."
  },
  {
    "id": "q15",
    "type": "qa",
    "topic": "Basic deep learning",
    "title": "Choose the normalization axis",
    "prompt": "A transformer activation has shape (batch, tokens, features). What does LayerNorm(features) normalize, and how does this differ from the usual behavior of BatchNorm?",
    "answer": [
      "LayerNorm(features) computes a mean and variance over the feature vector of each token independently; it does not pool statistics across examples or tokens. The usual BatchNorm computes per-channel statistics across the batch and applicable spatial or sequence dimensions during training, then uses running statistics during evaluation. The exact axes depend on the BatchNorm layer and tensor layout. LayerNorm does not rely on batch statistics, which makes its behavior convenient for variable batch sizes and token-wise transformer representations."
    ],
    "takeaway": "LayerNorm normalizes each token's feature vector; BatchNorm usually shares statistics across examples."
  },
  {
    "id": "q16",
    "type": "qa",
    "topic": "Basic deep learning",
    "title": "Why does an adaptive optimizer still need a learning rate?",
    "prompt": "Plain SGD has w = 1, gradient g = 0.2, and learning rate 0.1. What is the updated weight? Why should that learning rate not be copied blindly when switching to AdamW?",
    "answer": [
      "Without momentum or weight decay, SGD updates w to 1 − 0.1 × 0.2 = 0.98. AdamW uses running first and second gradient moments to change the direction and relative size of per-parameter updates, while applying weight decay separately. Its learning rate still controls the overall update scale. The same numeric learning rate can behave very differently across optimizers, so retune it and its schedule using training stability and validation performance."
    ],
    "takeaway": "Adaptive updates still have an overall step size that must be tuned."
  },
  {
    "id": "q17",
    "type": "qa",
    "topic": "Basic deep learning",
    "title": "Catch misleading accuracy on an imbalanced dataset",
    "prompt": "A dataset has 1,000 examples, including 10 positives. A classifier predicts negative for every example. Calculate its accuracy and positive-class recall. What should you inspect before calling it useful?",
    "answer": [
      "It correctly classifies 990 negatives, so accuracy is 99%. It finds none of the 10 positives, so positive-class recall is 0%. Inspect the confusion matrix, precision and recall across thresholds, and the cost of false positives versus false negatives. Precision has no positive predictions in its denominator here, so it is undefined mathematically; reporting conventions may set it to zero. High accuracy alone conceals complete failure to detect the rare class."
    ],
    "takeaway": "Judge rare-class performance with class-specific metrics and the decision costs."
  },
  {
    "id": "q18",
    "type": "qa",
    "topic": "Transformers",
    "title": "Track multi-head attention shapes",
    "prompt": "In standard self-attention, batch size is 2, sequence length is 5, model width is 768, and there are 12 heads. What are the per-head query shape, attention-score shape, and concatenated-head output shape?",
    "answer": [
      "Each head has width 768/12 = 64. After splitting and transposing, queries have shape (2, 12, 5, 64). Multiplying queries by transposed keys gives scores of shape (2, 12, 5, 5), with one query-key matrix per batch item and head. Weighted values retain shape (2, 12, 5, 64). Merging heads produces (2, 5, 768), which the output projection transforms back into the model's representation space."
    ],
    "takeaway": "Split feature width into heads, attend within each head, then merge the head features."
  },
  {
    "id": "q19",
    "type": "qa",
    "topic": "Transformers",
    "title": "What positional information does RoPE provide?",
    "prompt": "Why does self-attention need positional information? How does rotary positional embedding inject it into queries and keys, and does using RoPE guarantee reliable inference at much longer contexts?",
    "answer": [
      "Without positional signals or other order-sensitive structure, attention cannot distinguish token order from content alone. RoPE rotates pairs of query and key features by position-dependent angles. With fixed frequencies, their dot products encode relative positional offsets through the difference between rotations. This gives attention an order signal without simply adding a learned position vector. It does not guarantee reliable extrapolation: training lengths, frequency choices, scaling methods, and the model's learned behavior still matter at longer contexts."
    ],
    "takeaway": "RoPE makes query-key interactions position-dependent, while long-context reliability still needs validation."
  },
  {
    "id": "q20",
    "type": "qa",
    "topic": "Transformers",
    "title": "Compare pre-norm and post-norm residual blocks",
    "prompt": "Compare x + F(LayerNorm(x)) with LayerNorm(x + F(x)). Which is pre-norm, and why can the placement of normalization affect optimization in a deep transformer?",
    "answer": [
      "The first expression is pre-norm: normalize the input to the sublayer, then add its output to x. The second is post-norm: normalize after the residual addition. Pre-norm preserves a direct identity path through the residual stream, which often helps gradients travel through deep stacks. Post-norm passes that residual path through normalization. This changes optimization behavior, but neither layout guarantees better quality in every setting; initialization, depth, and training choices also matter."
    ],
    "takeaway": "Normalization placement changes the gradient path through a residual stack."
  },
  {
    "id": "q21",
    "type": "qa",
    "topic": "Transformers",
    "title": "Identify where queries, keys and values come from",
    "prompt": "In a standard encoder-decoder transformer for sequence generation, compare encoder self-attention, decoder self-attention, and cross-attention. Which tokens provide Q, K, and V, and where is a causal mask needed?",
    "answer": [
      "Encoder self-attention takes Q, K, and V from the source representations and usually permits attention to every valid source token. Decoder self-attention takes all three from target representations and uses a causal mask to hide future target tokens. Cross-attention takes queries from the decoder and keys and values from encoder outputs, allowing each target position to consult the encoded source. Source padding must still be masked. These are the standard roles; particular architectures can use different attention patterns."
    ],
    "takeaway": "Self-attention reads one sequence; cross-attention uses queries from one sequence to read another."
  },
  {
    "id": "q22",
    "type": "qa",
    "topic": "Transformers",
    "title": "Combine causal and padding constraints",
    "prompt": "With attention rows as queries and columns as keys, why does a causal mask not replace a key-padding mask? For query i and key j, state when attention should be allowed, and explain what to do with padded query outputs.",
    "answer": [
      "A causal mask hides future keys, while a key-padding mask hides positions that do not contain real input tokens. For a valid query i, allow key j only when j is a valid token and j ≤ i. Padded query outputs are generally excluded from the loss or otherwise ignored; masking padding keys alone does not make those query outputs meaningful. Avoid applying an ordinary softmax to a row with every key masked, because it has no valid probability distribution. Boolean-mask conventions also differ between APIs."
    ],
    "takeaway": "Causality restricts time, padding restricts valid tokens, and padded queries need separate treatment."
  },
  {
    "id": "q23",
    "type": "qa",
    "topic": "Transformers",
    "title": "Why keep a feed-forward block after attention?",
    "prompt": "Attention already combines token representations. What does a transformer's position-wise feed-forward block add, and how does a SwiGLU-style gated block differ from a single linear projection?",
    "answer": [
      "Attention mixes information across token positions. The feed-forward block applies a learned nonlinear feature transformation independently at each position, giving the layer additional capacity to process what attention gathered. A SwiGLU-style block forms two projections: one passes through SiLU and gates the other through element-wise multiplication, followed by an output projection. This introduces nonlinear, input-dependent feature gating. Unlike attention, the position-wise block itself does not directly exchange information between different token positions."
    ],
    "takeaway": "Attention mixes tokens; the feed-forward block transforms each token's features nonlinearly."
  },
  {
    "id": "q24",
    "type": "qa",
    "topic": "Transformers",
    "title": "What is shared in grouped-query attention?",
    "prompt": "A model has 16 query heads but only 4 key/value heads. How are they grouped in grouped-query attention, and how does its KV-cache size compare with 16 key/value heads when everything else is equal?",
    "answer": [
      "Each key/value head is shared by a group of four query heads. The query heads retain separate queries and attention distributions, but members of a group read the same projected keys and values. Keeping head dimension, layers, batch size, context length, and cache precision fixed, storing four KV heads instead of sixteen uses one quarter of the KV-cache memory. This does not mean all attention computation falls by a factor of four, and changing the number of KV heads can affect model quality."
    ],
    "takeaway": "Grouped-query attention shares keys and values across query heads to reduce KV storage."
  },
  {
    "id": "q25",
    "type": "qa",
    "topic": "Transformers",
    "title": "Separate attention arithmetic from score storage",
    "prompt": "For dense self-attention with fixed batch size, head count, and head dimension, sequence length doubles from 1,024 to 2,048. How do pairwise score computation and a materialized score matrix scale? How can tiled attention reduce memory without making attention sparse?",
    "answer": [
      "Both the number of query-key pairs and the elements in a materialized score matrix grow approximately fourfold, because they scale with sequence length squared. Tiled implementations can compute blocks of scores and update softmax/output statistics without retaining the full matrix in memory. They still evaluate dense attention rather than discarding token pairs. Such implementations can reduce intermediate storage and memory traffic while preserving the same attention calculation up to floating-point differences; they do not remove its quadratic pairwise arithmetic."
    ],
    "takeaway": "Dense attention has quadratic pairwise work, but its full score matrix need not be stored."
  },
  {
    "id": "q26",
    "type": "qa",
    "topic": "LLM fine-tuning",
    "title": "Mask the loss, keep the conversation",
    "prompt": "For assistant-only supervised fine-tuning, a sequence contains user tokens, assistant tokens and padding. Which target labels should be ignored, and why should you keep the user tokens in the input?",
    "answer": [
      "Ignore the targets for user or other prompt tokens and padding, commonly with label value -100. Keep the prompt tokens as inputs: the assistant needs that context to predict its response. Compute loss on the intended assistant response tokens, including its end token if the training format calls for one. Apply the next-token shift consistently; some model APIs perform it internally. Average over supervised tokens rather than including ignored positions in the denominator."
    ],
    "takeaway": "Loss masking controls what the model learns to predict while retaining the context needed for that prediction."
  },
  {
    "id": "q27",
    "type": "qa",
    "topic": "LLM fine-tuning",
    "title": "Split instruction data without leakage",
    "prompt": "You generate ten question–answer examples from each source document, then randomly split all rows into training and validation. Why might validation look too good, and how would you repair the split?",
    "answer": [
      "Different examples from the same document can reveal the same facts or nearly identical wording across splits. The model may recognize training content rather than generalize to unseen sources. Group examples by source document before splitting, and check for exact and near duplicates across groups. Choose grouping appropriate to deployment, such as customer, conversation or time period. Keep the final test set separate from prompt, data and hyperparameter decisions."
    ],
    "takeaway": "Split by the underlying source of related examples, not merely by generated rows."
  },
  {
    "id": "q28",
    "type": "qa",
    "topic": "LLM fine-tuning",
    "title": "Keep chat formatting consistent",
    "prompt": "A chat model was fine-tuned on role markers and end-of-turn tokens, but inference concatenates plain user and assistant text. What can go wrong, and what formatting checks would you make?",
    "answer": [
      "The model may see an unfamiliar conversation structure, misread role boundaries or continue the wrong speaker. Use the appropriate tokenizer and chat template, with compatible role markers, turn endings and the assistant generation prefix. Inspect the rendered text and actual token IDs in both training and inference. If a template already inserts special tokens, avoid adding them a second time during tokenization. Check that response labels and stopping tokens match the chosen format."
    ],
    "takeaway": "Conversation formatting is part of the model’s learned input distribution."
  },
  {
    "id": "q29",
    "type": "qa",
    "topic": "LLM fine-tuning",
    "title": "Choose between SFT and preference training",
    "prompt": "One dataset contains ideal responses; another contains preferred and rejected responses for the same prompts. How do supervised fine-tuning and standard DPO use these datasets differently?",
    "answer": [
      "SFT trains next-token prediction on demonstrations of the behavior you want. Preference training uses comparisons to encourage the preferred response relative to the rejected one. Standard DPO uses paired responses and a reference policy in its objective, without separately fitting a reward model or collecting on-policy rollouts. Both depend on reliable, relevant data. Preference optimization often follows SFT when the starting model first needs basic instruction-following behavior."
    ],
    "takeaway": "Demonstrations teach a target response; preference pairs teach a relative choice between responses."
  },
  {
    "id": "q30",
    "type": "qa",
    "topic": "LLM fine-tuning",
    "title": "Separate QLoRA storage from computation",
    "prompt": "QLoRA stores frozen base weights in a low-bit representation. Does that mean adapter weights, gradients and every arithmetic operation must also use that bit width? Explain the distinction.",
    "answer": [
      "No. The frozen base weights use a compressed storage representation and are dequantized for computation, typically with a higher-precision compute dtype such as bfloat16. Trainable LoRA adapters and their gradients also use higher precision; optimizer state has its own storage requirements. Quantizing the base reduces its weight-storage cost, while activations, temporary buffers and adapter training still consume memory. End-to-end memory therefore does not shrink by the same ratio as base-weight storage."
    ],
    "takeaway": "A weight’s storage bit width and the precision used to compute with it are separate choices."
  },
  {
    "id": "q31",
    "type": "qa",
    "topic": "LLM fine-tuning",
    "title": "Calculate LoRA rank, scaling and merging",
    "prompt": "For a base weight W of shape (8, 16), use LoRA rank r=4 and alpha=8 with no adapter bias. How many adapter parameters are trained, what is the alpha/r scale, and how do you merge the adapter?",
    "answer": [
      "A has shape (4, 16) and B has shape (8, 4), so there are 4×16 + 8×4 = 96 trainable adapter parameters. The conventional alpha/r scale is 8/4 = 2, and the merged weight is W + 2BA. This parameter count excludes the frozen base weight. Raising rank changes capacity and, if alpha stays fixed, changes this scaling factor; it does not guarantee a better model or a particular update magnitude."
    ],
    "takeaway": "Count both low-rank matrices and apply the same adapter scaling when merging."
  },
  {
    "id": "q32",
    "type": "qa",
    "topic": "LLM fine-tuning",
    "title": "Choose the right training-memory tool",
    "prompt": "Your intended batch is too large for GPU memory. What do gradient accumulation and activation checkpointing each change, and do either remove optimizer state?",
    "answer": [
      "Gradient accumulation processes smaller microbatches and accumulates gradients before an optimizer step, reducing the activation memory needed for the intended effective batch. Normalize losses to preserve the intended batch or token weighting. Activation checkpointing saves fewer intermediate activations and recomputes them during backward, trading extra compute for memory. Neither removes optimizer state. If even one microbatch does not fit, accumulation alone is insufficient; checkpointing or another memory reduction is needed."
    ],
    "takeaway": "Accumulation reduces simultaneous batch work; checkpointing trades recomputation for saved activations."
  },
  {
    "id": "q33",
    "type": "qa",
    "topic": "LLM inference",
    "title": "Distinguish temperature from nucleus size",
    "prompt": "When temperature is applied before top-p filtering, can changing a positive temperature change which tokens enter the nucleus even though their logit ranking stays the same? How does top-k differ?",
    "answer": [
      "Yes. Dividing logits by a positive temperature preserves their ranking but changes their normalized probabilities. Top-p keeps the smallest high-probability prefix reaching a cumulative-probability threshold, so a flatter distribution can require more tokens. Top-k keeps a fixed number of highest-ranked tokens; their membership is unchanged by positive temperature scaling when tie handling is fixed. Sampling probabilities within either retained set still depend on temperature."
    ],
    "takeaway": "Temperature changes probability mass; top-p limits mass and top-k limits the number of candidates."
  },
  {
    "id": "q34",
    "type": "qa",
    "topic": "LLM inference",
    "title": "Budget a grouped-query KV cache",
    "prompt": "A model has 32 layers, 32 query heads, 8 KV heads and head dimension 128. For a batch of 2 sequences, each caching 4096 tokens in a 2-byte dtype, how much memory does a dense KV cache require? Ignore overhead.",
    "answer": [
      "Use 2 × layers × batch × tokens × KV heads × head dimension × bytes per element. The leading 2 accounts for keys and values. Here this is 2×32×2×4096×8×128×2 = 1,073,741,824 bytes, or 1 GiB. Use the 8 KV heads rather than the 32 query heads. This estimate excludes model weights, activations, allocation overhead and extra padding or cache capacity reserved by the implementation."
    ],
    "takeaway": "KV-cache memory depends on stored KV heads, not the number of query heads."
  },
  {
    "id": "q35",
    "type": "qa",
    "topic": "LLM inference",
    "title": "Measure the waiting time users notice",
    "prompt": "A user waits three seconds for the first token, then receives one token every 20 milliseconds. Which latency metrics describe these two experiences, and why does aggregate tokens per second miss part of the problem?",
    "answer": [
      "Time to first token (TTFT) measures the initial wait and can include queueing, prompt processing, first-token generation and transport, depending on the measurement boundary. Inter-token latency measures the gaps during streaming; 20 ms corresponds to about 50 tokens per second for that stream after the first token. Aggregate throughput across many requests can look high while individual users still experience slow starts or long gaps. Measure both latency and throughput."
    ],
    "takeaway": "Fast aggregate throughput does not establish low initial or streaming latency for an individual request."
  },
  {
    "id": "q36",
    "type": "qa",
    "topic": "LLM inference",
    "title": "Batch by cache demand, not just request count",
    "prompt": "Doubling a serving batch improves throughput on short prompts but causes out-of-memory failures when prompts are long. Why is a request-count limit insufficient, and what budget would you add?",
    "answer": [
      "Each active request consumes KV-cache space proportional to its stored token count, so equal-size batches can have very different memory demands. Add a budget for total cached tokens or estimated cache bytes, accounting for expected output growth and implementation overhead. Larger batches may improve hardware utilization, but do not guarantee lower per-request latency. Admission and scheduling should respect both memory and latency budgets; continuous batching can replace finished requests without waiting for the entire batch."
    ],
    "takeaway": "Serving capacity depends on how many tokens active requests retain, as well as how many requests are active."
  },
  {
    "id": "q37",
    "type": "qa",
    "topic": "LLM inference",
    "title": "Check what inference quantization actually saves",
    "prompt": "Replacing 16-bit model weights with a 4-bit representation reduces weight storage substantially. Why might latency improve little, and which other memory costs and quality checks remain?",
    "answer": [
      "Speed depends on compatible kernels, hardware, dequantization overhead and whether the workload is limited by memory bandwidth or computation. A smaller representation alone does not guarantee faster execution. Scales and metadata add storage overhead, while activations and the KV cache may remain at higher precision. Measure latency and memory for the actual batch and context sizes, and evaluate output quality on representative tasks because quantization errors can affect sensitive weights and behaviors."
    ],
    "takeaway": "Benchmark the complete quantized workload rather than inferring speed or quality from weight bit width."
  },
  {
    "id": "q38",
    "type": "qa",
    "topic": "LLM inference",
    "title": "Verify speculative sampling exactly",
    "prompt": "In standard exact speculative sampling, a token sampled from draft distribution q has q(token)=0.8 and target distribution p has p(token)=0.4 at the same prefix. What is its acceptance probability, and how does rejection preserve the target distribution?",
    "answer": [
      "Accept it with probability min(1, p/q) = 0.5. On rejection, sample from the normalized positive residual max(p−q, 0), rather than simply accepting a token with a high raw target probability. This correction preserves the target sampling distribution when p and q are the actual distributions at that prefix, including any sampling transformations. Exactness depends on correct verification and correction; merely allowing plausible draft tokens can change the output distribution."
    ],
    "takeaway": "Exact speculative sampling uses the target-to-draft probability ratio and a corrected rejection distribution."
  },
  {
    "id": "q39",
    "type": "qa",
    "topic": "LLM inference",
    "title": "Fit generated tokens into the context budget",
    "prompt": "After chat formatting, a prompt contains 4000 tokens. The supported total context budget is 4096 tokens, and max_new_tokens is 128. Without truncation or context extension, what output budget fits, and what might stop generation earlier?",
    "answer": [
      "Only 4096−4000 = 96 additional tokens fit under that total-context assumption, so configure an output budget no larger than 96. The prompt count must include template and special tokens. Generation may finish sooner if the model emits a configured EOS token or matches a stopping rule. Token limits count tokenizer tokens rather than characters or words, and some serving APIs impose additional limits or reject requests whose requested budget exceeds capacity."
    ],
    "takeaway": "Budget prompt and generated tokens together, and treat the output limit as a maximum rather than a target length."
  },
  {
    "id": "q40",
    "type": "coding",
    "topic": "Basic deep learning",
    "title": "Match a full batch with gradient accumulation",
    "prompt": "Implement one SGD update using a list of (inputs, targets) microbatches of unequal sizes. The model has one output per example and uses mean squared error. Zero gradients once, weight each microbatch mean loss by its fraction of the total examples, backpropagate, then step once. Compare the result with a single full-batch update.",
    "answer": [
      "Gradients add across backward calls. Multiply each microbatch mean loss by its example count divided by the total count; otherwise unequal microbatches receive equal weight. Clear gradients before the accumulation window and call the optimizer after all its microbatches. This equivalence assumes identical per-example computation, with no batch-dependent layers or stochastic differences."
    ],
    "takeaway": "Accumulate weighted microbatch gradients and perform one optimizer step per full accumulation window.",
    "starterCode": "import copy\nimport torch\nimport torch.nn.functional as F\n\ndef accumulated_step(model, optimizer, microbatches):\n    # TODO: implement the requested operation.\n    raise NotImplementedError\n\nmodel = torch.nn.Linear(2, 1, bias=False)\nwith torch.no_grad():\n    model.weight.copy_(torch.tensor([[0.2, -0.3]]))\nreference = copy.deepcopy(model)\nx = torch.tensor([[1., 2.], [3., 4.], [-1., 1.]])\ny = torch.tensor([[1.], [0.], [2.]])\noptimizer = torch.optim.SGD(model.parameters(), lr=0.1)\nreference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)\nreference_optimizer.zero_grad()\nF.mse_loss(reference(x), y).backward()\nreference_optimizer.step()\naccumulated_step(model, optimizer, [(x[:2], y[:2]), (x[2:], y[2:])])\nassert torch.allclose(model.weight, reference.weight, atol=1e-7)\nassert torch.allclose(model.weight, torch.tensor([[0.2466667, 0.2133333]]), atol=1e-6)\nprint(model.weight.detach())\n",
    "solutionCode": "import copy\nimport torch\nimport torch.nn.functional as F\n\ndef accumulated_step(model, optimizer, microbatches):\n    total = sum(x.shape[0] for x, _ in microbatches)\n    optimizer.zero_grad()\n    for x, target in microbatches:\n        loss = F.mse_loss(model(x), target)\n        (loss * (x.shape[0] / total)).backward()\n    optimizer.step()\n\nmodel = torch.nn.Linear(2, 1, bias=False)\nwith torch.no_grad():\n    model.weight.copy_(torch.tensor([[0.2, -0.3]]))\nreference = copy.deepcopy(model)\nx = torch.tensor([[1., 2.], [3., 4.], [-1., 1.]])\ny = torch.tensor([[1.], [0.], [2.]])\noptimizer = torch.optim.SGD(model.parameters(), lr=0.1)\nreference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)\nreference_optimizer.zero_grad()\nF.mse_loss(reference(x), y).backward()\nreference_optimizer.step()\naccumulated_step(model, optimizer, [(x[:2], y[:2]), (x[2:], y[2:])])\nassert torch.allclose(model.weight, reference.weight, atol=1e-7)\nassert torch.allclose(model.weight, torch.tensor([[0.2466667, 0.2133333]]), atol=1e-6)\nprint(model.weight.detach())\n",
    "checks": [
      "Unequal microbatches produce the same weights as the full-batch reference.",
      "The updated weight is approximately [0.246667, 0.213333]."
    ]
  },
  {
    "id": "q41",
    "type": "coding",
    "topic": "Basic deep learning",
    "title": "Clip the global gradient norm",
    "prompt": "Implement global L2 gradient clipping using PyTorch. Return the norm before clipping. The L2 norm must combine gradients from every supplied parameter, rather than clipping each parameter independently. A gradient vector with norm 5 should be reduced to norm 2; smaller gradients should stay unchanged.",
    "answer": [
      "clip_grad_norm_ computes the norm across all supplied parameter gradients and rescales them together when it exceeds max_norm. It modifies gradients in place and returns the original norm. Call it after backward and before the optimizer step. With mixed-precision gradient scaling, unscale the gradients before clipping them."
    ],
    "takeaway": "Clip after backward and before the optimizer step, using one norm across the parameter set.",
    "starterCode": "import torch\n\ndef clip_gradients(parameters, max_norm):\n    # TODO: implement the requested operation.\n    raise NotImplementedError\n\np = torch.nn.Parameter(torch.zeros(2))\nq = torch.nn.Parameter(torch.zeros(1))\np.grad = torch.tensor([3., 0.])\nq.grad = torch.tensor([4.])\nbefore = clip_gradients([p, q], 2.0)\nassert torch.allclose(before, torch.tensor(5.))\nassert torch.allclose(p.grad, torch.tensor([1.2, 0.]), atol=1e-5)\nassert torch.allclose(q.grad, torch.tensor([1.6]), atol=1e-5)\nassert torch.linalg.vector_norm(torch.cat([p.grad, q.grad])) <= 2.000001\np.grad = torch.tensor([0.3, 0.4])\nq.grad = torch.tensor([0.])\nclip_gradients([p, q], 2.0)\nassert torch.equal(p.grad, torch.tensor([0.3, 0.4]))\nprint('Global clipping checks passed.')\n",
    "solutionCode": "import torch\n\ndef clip_gradients(parameters, max_norm):\n    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)\n\np = torch.nn.Parameter(torch.zeros(2))\nq = torch.nn.Parameter(torch.zeros(1))\np.grad = torch.tensor([3., 0.])\nq.grad = torch.tensor([4.])\nbefore = clip_gradients([p, q], 2.0)\nassert torch.allclose(before, torch.tensor(5.))\nassert torch.allclose(p.grad, torch.tensor([1.2, 0.]), atol=1e-5)\nassert torch.allclose(q.grad, torch.tensor([1.6]), atol=1e-5)\nassert torch.linalg.vector_norm(torch.cat([p.grad, q.grad])) <= 2.000001\np.grad = torch.tensor([0.3, 0.4])\nq.grad = torch.tensor([0.])\nclip_gradients([p, q], 2.0)\nassert torch.equal(p.grad, torch.tensor([0.3, 0.4]))\nprint('Global clipping checks passed.')\n",
    "checks": [
      "The returned pre-clipping norm is 5.",
      "The global norm spans both parameters: [3, 0] and [4] become approximately [1.2, 0] and [1.6].",
      "Gradients already below the threshold remain unchanged."
    ]
  },
  {
    "id": "q42",
    "type": "coding",
    "topic": "Basic deep learning",
    "title": "Pool token embeddings without counting padding",
    "prompt": "Implement masked mean pooling for finite embeddings of shape (batch, tokens, hidden) and a Boolean valid-token mask of shape (batch, tokens). Average only valid tokens. Define an all-padding sequence to return a zero vector. Verify that padding values and their gradients cannot affect the pooled result.",
    "answer": [
      "Expand the token mask across the hidden dimension, sum the masked embeddings, and divide by the valid-token count. Clamp a zero count to one to implement the stated zero-vector convention for an all-padding sequence. Masked positions contribute neither values nor gradients to this pooling operation."
    ],
    "takeaway": "The denominator counts valid tokens, not the padded sequence length.",
    "starterCode": "import torch\n\ndef masked_mean(hidden, valid_tokens):\n    # TODO: implement the requested operation.\n    raise NotImplementedError\n\nhidden = torch.tensor([[[1., 2.], [3., 4.], [999., 999.]],\n                       [[7., 8.], [9., 10.], [11., 12.]]], requires_grad=True)\nvalid = torch.tensor([[True, True, False], [False, False, False]])\npooled = masked_mean(hidden, valid)\nassert torch.equal(pooled, torch.tensor([[2., 3.], [0., 0.]]))\npooled.sum().backward()\nassert torch.equal(hidden.grad[~valid], torch.zeros_like(hidden.grad[~valid]))\nassert torch.allclose(hidden.grad[0, :2], torch.full((2, 2), 0.5))\nprint(pooled.detach())\n",
    "solutionCode": "import torch\n\ndef masked_mean(hidden, valid_tokens):\n    weights = valid_tokens.unsqueeze(-1).to(hidden.dtype)\n    total = (hidden * weights).sum(dim=1)\n    count = weights.sum(dim=1).clamp_min(1)\n    return total / count\n\nhidden = torch.tensor([[[1., 2.], [3., 4.], [999., 999.]],\n                       [[7., 8.], [9., 10.], [11., 12.]]], requires_grad=True)\nvalid = torch.tensor([[True, True, False], [False, False, False]])\npooled = masked_mean(hidden, valid)\nassert torch.equal(pooled, torch.tensor([[2., 3.], [0., 0.]]))\npooled.sum().backward()\nassert torch.equal(hidden.grad[~valid], torch.zeros_like(hidden.grad[~valid]))\nassert torch.allclose(hidden.grad[0, :2], torch.full((2, 2), 0.5))\nprint(pooled.detach())\n",
    "checks": [
      "The first sequence pools to [2, 3]; the all-padding sequence pools to zero.",
      "Padded positions have zero gradient.",
      "Each valid token receives a gradient of 0.5 for each hidden component."
    ]
  },
  {
    "id": "q43",
    "type": "coding",
    "topic": "Transformers",
    "title": "Split and merge attention heads",
    "prompt": "Complete split_heads to transform (batch, tokens, model_dim) into (batch, heads, tokens, head_dim). model_dim is divisible by heads. Preserve token order and each head’s feature slice. The merge_heads helper reverses the operation; use its round-trip and gradient checks to validate the split.",
    "answer": [
      "Reshape the last feature dimension into heads and head_dim, then transpose the token and head axes. To merge, transpose those axes back and reshape the adjacent head dimensions into model_dim. Calling contiguous before the final reshape makes the intended memory layout explicit after the transpose."
    ],
    "takeaway": "Splitting heads changes both shape and axis order; merging must reverse both operations.",
    "starterCode": "import torch\n\ndef split_heads(x, heads):\n    # TODO: implement the requested operation.\n    raise NotImplementedError\n\ndef merge_heads(x):\n    batch, heads, tokens, head_dim = x.shape\n    return x.transpose(1, 2).contiguous().reshape(batch, tokens, heads * head_dim)\n\nx = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8).requires_grad_()\nheads = split_heads(x, 2)\nassert heads.shape == (2, 2, 3, 4)\nassert torch.equal(heads[0, 1, 2], x[0, 2, 4:])\nmerged = merge_heads(heads)\nassert torch.equal(merged, x)\nmerged.sum().backward()\nassert torch.equal(x.grad, torch.ones_like(x))\nprint('Head layout round trip passed.')\n",
    "solutionCode": "import torch\n\ndef split_heads(x, heads):\n    batch, tokens, model_dim = x.shape\n    return x.reshape(batch, tokens, heads, model_dim // heads).transpose(1, 2)\n\ndef merge_heads(x):\n    batch, heads, tokens, head_dim = x.shape\n    return x.transpose(1, 2).contiguous().reshape(batch, tokens, heads * head_dim)\n\nx = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8).requires_grad_()\nheads = split_heads(x, 2)\nassert heads.shape == (2, 2, 3, 4)\nassert torch.equal(heads[0, 1, 2], x[0, 2, 4:])\nmerged = merge_heads(heads)\nassert torch.equal(merged, x)\nmerged.sum().backward()\nassert torch.equal(x.grad, torch.ones_like(x))\nprint('Head layout round trip passed.')\n",
    "checks": [
      "The split shape is (2, 2, 3, 4).",
      "Head 1 at token 2 contains the original feature slice 4..7.",
      "Merging reproduces every input value and preserves gradients."
    ]
  },
  {
    "id": "q44",
    "type": "coding",
    "topic": "Transformers",
    "title": "Apply rotary positions to adjacent feature pairs",
    "prompt": "Implement RoPE for x of shape (tokens, head_dim), with an even head_dim and a position vector of length tokens. Rotate adjacent pairs (x0,x1), (x2,x3), using frequencies base^(-2j/head_dim). Support nonzero position offsets. Use the adjacent-pair convention explicitly; some model implementations arrange their pairs differently.",
    "answer": [
      "Each feature pair undergoes a two-dimensional rotation by position times its inverse frequency. The rotation preserves the pair’s norm and is the identity at position zero. Explicit positions let cached decoding continue at the correct offset instead of restarting at zero for each new token. Pairing conventions must match the model being implemented."
    ],
    "takeaway": "RoPE is a position-dependent rotation; cached tokens need their actual positions.",
    "starterCode": "import math\nimport torch\n\ndef apply_rope(x, positions, base=10000.0):\n    # TODO: implement the requested operation.\n    raise NotImplementedError\n\nx = torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.]], dtype=torch.float64)\npositions = torch.tensor([3, 4])\ny = apply_rope(x, positions)\nexpected_first = torch.tensor([math.cos(3), math.sin(3),\n                               math.cos(0.03), math.sin(0.03)], dtype=torch.float64)\nassert torch.allclose(y[0], expected_first, atol=1e-12)\nassert torch.allclose(torch.linalg.vector_norm(y, dim=-1),\n                      torch.linalg.vector_norm(x, dim=-1), atol=1e-12)\nassert torch.equal(apply_rope(x, torch.zeros(2)), x)\nprint(y)\n",
    "solutionCode": "import math\nimport torch\n\ndef apply_rope(x, positions, base=10000.0):\n    dim = x.shape[-1]\n    frequency = base ** (-torch.arange(0, dim, 2, dtype=x.dtype, device=x.device) / dim)\n    angles = positions.to(dtype=x.dtype, device=x.device).unsqueeze(-1) * frequency\n    cosine, sine = angles.cos(), angles.sin()\n    pairs = x.reshape(x.shape[0], dim // 2, 2)\n    even, odd = pairs[..., 0], pairs[..., 1]\n    rotated = torch.stack((even * cosine - odd * sine,\n                           even * sine + odd * cosine), dim=-1)\n    return rotated.flatten(-2)\n\nx = torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.]], dtype=torch.float64)\npositions = torch.tensor([3, 4])\ny = apply_rope(x, positions)\nexpected_first = torch.tensor([math.cos(3), math.sin(3),\n                               math.cos(0.03), math.sin(0.03)], dtype=torch.float64)\nassert torch.allclose(y[0], expected_first, atol=1e-12)\nassert torch.allclose(torch.linalg.vector_norm(y, dim=-1),\n                      torch.linalg.vector_norm(x, dim=-1), atol=1e-12)\nassert torch.equal(apply_rope(x, torch.zeros(2)), x)\nprint(y)\n",
    "checks": [
      "Position 3 produces rotation angles 3 and 0.03 for a four-dimensional head.",
      "Rotations preserve the per-token norm.",
      "Position zero leaves the embedding unchanged."
    ]
  },
  {
    "id": "q45",
    "type": "coding",
    "topic": "Transformers",
    "title": "Combine causal and key-padding masks",
    "prompt": "Build a Boolean allowed-attention mask of shape (batch, 1, tokens, tokens) from a Boolean valid-key mask of shape (batch, tokens). True means allowed. A query may attend only to non-padding keys at its own or an earlier position. Assume nonempty right-padded sequences. This masks keys; padded query outputs are discarded separately.",
    "answer": [
      "The lower triangle allows keys at or before each query. Broadcasting the valid-key mask over query rows removes padding keys as well. Logical AND combines the two conditions. This Boolean convention is explicitly True for allowed attention; some APIs instead use True for positions to block, so their conventions need checking."
    ],
    "takeaway": "Causal masking constrains time; key-padding masking constrains which keys exist.",
    "starterCode": "import torch\n\ndef allowed_attention(valid_keys):\n    # TODO: implement the requested operation.\n    raise NotImplementedError\n\nvalid = torch.tensor([[True, True, False, False],\n                      [True, True, True, False]])\nallowed = allowed_attention(valid)\nexpected = torch.tensor([[[[1, 0, 0, 0], [1, 1, 0, 0],\n                           [1, 1, 0, 0], [1, 1, 0, 0]]],\n                         [[[1, 0, 0, 0], [1, 1, 0, 0],\n                           [1, 1, 1, 0], [1, 1, 1, 0]]]], dtype=torch.bool)\nassert allowed.shape == (2, 1, 4, 4)\nassert torch.equal(allowed, expected)\nassert not allowed[..., -1].any()\nassert not allowed.triu(1).any()\nprint(allowed.to(torch.int))\n",
    "solutionCode": "import torch\n\ndef allowed_attention(valid_keys):\n    tokens = valid_keys.shape[1]\n    positions = torch.arange(tokens, device=valid_keys.device)\n    causal = positions[:, None] >= positions[None, :]\n    return causal[None, None, :, :] & valid_keys[:, None, None, :]\n\nvalid = torch.tensor([[True, True, False, False],\n                      [True, True, True, False]])\nallowed = allowed_attention(valid)\nexpected = torch.tensor([[[[1, 0, 0, 0], [1, 1, 0, 0],\n                           [1, 1, 0, 0], [1, 1, 0, 0]]],\n                         [[[1, 0, 0, 0], [1, 1, 0, 0],\n                           [1, 1, 1, 0], [1, 1, 1, 0]]]], dtype=torch.bool)\nassert allowed.shape == (2, 1, 4, 4)\nassert torch.equal(allowed, expected)\nassert not allowed[..., -1].any()\nassert not allowed.triu(1).any()\nprint(allowed.to(torch.int))\n",
    "checks": [
      "The mask matches the explicit expected lower-triangular valid-key pattern.",
      "No query may attend to the final padding key.",
      "Every future-key entry is False, while valid diagonal entries are True."
    ]
  },
  {
    "id": "q46",
    "type": "coding",
    "topic": "LLM fine-tuning",
    "title": "Compute an assistant-only next-token loss",
    "prompt": "Implement a loss from logits (batch, tokens, vocabulary), input token IDs, an assistant-token mask, and a valid-token mask. Masks describe target-token positions. Shift logits and targets for next-token prediction, then ignore user tokens and padding using label -100. Assume at least one supervised target remains. This toy function handles the shift itself; do not shift again in a model that already handles it.",
    "answer": [
      "Logits at position t predict the token at t+1, so both token IDs and their supervision masks are sliced from position 1 onward. Set unsupervised target labels to -100 and use cross_entropy with ignore_index=-100. A loss mask decides which predictions train the model; it does not replace the attention mask controlling which input tokens are visible."
    ],
    "takeaway": "Shift target tokens and target-position masks together, then average only supervised predictions.",
    "starterCode": "import math\nimport torch\nimport torch.nn.functional as F\n\ndef assistant_only_loss(logits, input_ids, assistant_tokens, valid_tokens):\n    # TODO: implement the requested operation.\n    raise NotImplementedError\n\nids = torch.tensor([[0, 1, 2, 0]])\nassistant = torch.tensor([[False, False, True, True]])\nvalid = torch.tensor([[True, True, True, False]])\nlogits = torch.tensor([[[7., 0., 0.], [0., 0., math.log(2)],\n                        [1., 2., 3.], [0., 0., 0.]]], requires_grad=True)\nloss = assistant_only_loss(logits, ids, assistant, valid)\nassert torch.allclose(loss, torch.tensor(math.log(2)), atol=1e-6)\nloss.backward()\nassert torch.count_nonzero(logits.grad[0, 1]).item() > 0\nassert torch.equal(logits.grad[0, [0, 2, 3]], torch.zeros(3, 3))\nprint(f'Assistant-only loss = {loss.item():.6f}')\n",
    "solutionCode": "import math\nimport torch\nimport torch.nn.functional as F\n\ndef assistant_only_loss(logits, input_ids, assistant_tokens, valid_tokens):\n    targets = input_ids[:, 1:].clone()\n    supervised = assistant_tokens[:, 1:] & valid_tokens[:, 1:]\n    targets = targets.masked_fill(~supervised, -100)\n    predictions = logits[:, :-1, :]\n    return F.cross_entropy(predictions.reshape(-1, logits.shape[-1]),\n                           targets.reshape(-1), ignore_index=-100)\n\nids = torch.tensor([[0, 1, 2, 0]])\nassistant = torch.tensor([[False, False, True, True]])\nvalid = torch.tensor([[True, True, True, False]])\nlogits = torch.tensor([[[7., 0., 0.], [0., 0., math.log(2)],\n                        [1., 2., 3.], [0., 0., 0.]]], requires_grad=True)\nloss = assistant_only_loss(logits, ids, assistant, valid)\nassert torch.allclose(loss, torch.tensor(math.log(2)), atol=1e-6)\nloss.backward()\nassert torch.count_nonzero(logits.grad[0, 1]).item() > 0\nassert torch.equal(logits.grad[0, [0, 2, 3]], torch.zeros(3, 3))\nprint(f'Assistant-only loss = {loss.item():.6f}')\n",
    "checks": [
      "Only the prediction of token 2 is supervised in the example.",
      "The loss is log(2), approximately 0.693147.",
      "User-token, padding-target and final unused logits receive zero gradient."
    ]
  },
  {
    "id": "q47",
    "type": "coding",
    "topic": "LLM fine-tuning",
    "title": "Merge a scaled LoRA update",
    "prompt": "Implement a non-mutating LoRA merge for an ordinary floating-point linear weight W (out, in), A (rank, in), B (out, rank), and alpha. Use the conventional scale alpha/rank. Verify that a merged linear layer equals the base layer plus the separately computed adapter. Ignore bias and quantized-weight merging in this exercise.",
    "answer": [
      "The effective weight is W + (alpha/rank) BA. Merging stores this update in a new dense weight so a separate adapter branch is unnecessary at inference. The reference checks both the scaling and matrix order. Floating-point rounding can make real merged and unmerged models differ slightly; quantized bases may require additional dequantization or re-quantization steps."
    ],
    "takeaway": "With conventional LoRA scaling, merge W + (alpha/r) BA without altering the original base tensor.",
    "starterCode": "import torch\nimport torch.nn.functional as F\n\ndef merge_lora(W, A, B, alpha):\n    # TODO: implement the requested operation.\n    raise NotImplementedError\n\nW = torch.tensor([[1., 2.], [3., 4.]])\nA = torch.tensor([[1., 0.], [0., 2.]])\nB = torch.tensor([[1., 3.], [-2., 1.]])\nW_before = W.clone()\nalpha = 4.0\nmerged = merge_lora(W, A, B, alpha)\nassert torch.equal(merged, torch.tensor([[3., 14.], [-1., 8.]]))\nx = torch.tensor([[1., 2.], [-1., 1.]])\nunmerged = F.linear(x, W) + (alpha / A.shape[0]) * F.linear(F.linear(x, A), B)\nassert torch.allclose(F.linear(x, merged), unmerged, atol=1e-6)\nassert torch.equal(W, W_before)\nassert torch.equal(merge_lora(W, A, torch.zeros_like(B), alpha), W)\nprint(merged)\n",
    "solutionCode": "import torch\nimport torch.nn.functional as F\n\ndef merge_lora(W, A, B, alpha):\n    return W + (alpha / A.shape[0]) * (B @ A)\n\nW = torch.tensor([[1., 2.], [3., 4.]])\nA = torch.tensor([[1., 0.], [0., 2.]])\nB = torch.tensor([[1., 3.], [-2., 1.]])\nW_before = W.clone()\nalpha = 4.0\nmerged = merge_lora(W, A, B, alpha)\nassert torch.equal(merged, torch.tensor([[3., 14.], [-1., 8.]]))\nx = torch.tensor([[1., 2.], [-1., 1.]])\nunmerged = F.linear(x, W) + (alpha / A.shape[0]) * F.linear(F.linear(x, A), B)\nassert torch.allclose(F.linear(x, merged), unmerged, atol=1e-6)\nassert torch.equal(W, W_before)\nassert torch.equal(merge_lora(W, A, torch.zeros_like(B), alpha), W)\nprint(merged)\n",
    "checks": [
      "For alpha=4 and rank=2, the merged weight is [[3, 14], [-1, 8]].",
      "Merged and separate-adapter predictions agree.",
      "The original base weight stays unchanged, and a zero adapter changes nothing."
    ]
  },
  {
    "id": "q48",
    "type": "coding",
    "topic": "LLM fine-tuning",
    "title": "Separate AdamW decay and no-decay parameters",
    "prompt": "Build AdamW parameter groups for a toy Linear/LayerNorm model. Apply weight decay to trainable linear weight matrices; exclude all biases and LayerNorm parameters. Skip frozen parameters. Each trainable parameter must occur in exactly one group. This is the specified exercise policy, rather than a universal rule for every architecture.",
    "answer": [
      "Inspect parameters directly owned by each module to identify biases and LayerNorm parameters. Collect trainable parameters into disjoint decay and no-decay groups, and pass a zero weight_decay value for the latter. With zero gradients, AdamW still shrinks parameters in the decay group because its decay is decoupled from the gradient update."
    ],
    "takeaway": "Explicit optimizer groups make the chosen decay policy visible and prevent frozen or duplicate parameters from being optimized.",
    "starterCode": "import torch\n\ndef optimizer_groups(model, weight_decay):\n    # TODO: implement the requested operation.\n    raise NotImplementedError\n\ntorch.manual_seed(0)\nmodel = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.LayerNorm(3),\n                            torch.nn.Linear(3, 2))\nmodel[2].bias.requires_grad_(False)\ngroups = optimizer_groups(model, 0.1)\ndecay_ids = {id(p) for p in groups[0]['params']}\nno_decay_ids = {id(p) for p in groups[1]['params']}\nassert decay_ids == {id(model[0].weight), id(model[2].weight)}\nassert not decay_ids & no_decay_ids\ntrainable_ids = {id(p) for p in model.parameters() if p.requires_grad}\nassert decay_ids | no_decay_ids == trainable_ids\nassert sum(len(group['params']) for group in groups) == len(trainable_ids)\nbefore = {id(p): p.detach().clone() for p in model.parameters()}\noptimizer = torch.optim.AdamW(groups, lr=0.1)\nfor p in model.parameters():\n    if p.requires_grad:\n        p.grad = torch.zeros_like(p)\noptimizer.step()\nfor p in model.parameters():\n    expected = before[id(p)] * (0.99 if id(p) in decay_ids else 1.0)\n    assert torch.allclose(p, expected, atol=1e-6)\nprint('Decay groups and zero-gradient update passed.')\n",
    "solutionCode": "import torch\n\ndef optimizer_groups(model, weight_decay):\n    no_decay_ids = set()\n    for module in model.modules():\n        for name, parameter in module.named_parameters(recurse=False):\n            if name == 'bias' or isinstance(module, torch.nn.LayerNorm):\n                no_decay_ids.add(id(parameter))\n    decay, no_decay = [], []\n    for parameter in model.parameters():\n        if parameter.requires_grad:\n            (no_decay if id(parameter) in no_decay_ids else decay).append(parameter)\n    return [{'params': decay, 'weight_decay': weight_decay},\n            {'params': no_decay, 'weight_decay': 0.0}]\n\ntorch.manual_seed(0)\nmodel = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.LayerNorm(3),\n                            torch.nn.Linear(3, 2))\nmodel[2].bias.requires_grad_(False)\ngroups = optimizer_groups(model, 0.1)\ndecay_ids = {id(p) for p in groups[0]['params']}\nno_decay_ids = {id(p) for p in groups[1]['params']}\nassert decay_ids == {id(model[0].weight), id(model[2].weight)}\nassert not decay_ids & no_decay_ids\ntrainable_ids = {id(p) for p in model.parameters() if p.requires_grad}\nassert decay_ids | no_decay_ids == trainable_ids\nassert sum(len(group['params']) for group in groups) == len(trainable_ids)\nbefore = {id(p): p.detach().clone() for p in model.parameters()}\noptimizer = torch.optim.AdamW(groups, lr=0.1)\nfor p in model.parameters():\n    if p.requires_grad:\n        p.grad = torch.zeros_like(p)\noptimizer.step()\nfor p in model.parameters():\n    expected = before[id(p)] * (0.99 if id(p) in decay_ids else 1.0)\n    assert torch.allclose(p, expected, atol=1e-6)\nprint('Decay groups and zero-gradient update passed.')\n",
    "checks": [
      "Only the two trainable linear weight matrices receive decay.",
      "Groups are disjoint and cover each trainable parameter exactly once.",
      "A zero-gradient AdamW step shrinks decayed weights by 0.99 while excluded parameters stay unchanged."
    ]
  },
  {
    "id": "q49",
    "type": "coding",
    "topic": "LLM inference",
    "title": "Retain the threshold-crossing token in top-p sampling",
    "prompt": "Implement top_p_probs for finite CPU logits and 0 < top_p <= 1. Sort probabilities in descending order, keep the smallest prefix whose cumulative mass reaches top_p, and return a renormalized distribution in original vocabulary order. The sampling helper is provided. Keep the token that crosses the threshold, rather than discarding it.",
    "answer": [
      "A token belongs to the nucleus when the cumulative mass before it is still below top_p. Shift the cumulative sum by one position and always retain the first token. Then zero the excluded probabilities, map values back to original vocabulary indices, and renormalize. Sampling from that result gives the top-p distribution, which adapts its token count to the confidence of the logits."
    ],
    "takeaway": "Top-p keeps the threshold-crossing token and renormalizes the retained probability mass.",
    "starterCode": "import torch\n\ndef top_p_probs(logits, top_p):\n    # TODO: implement the requested operation.\n    raise NotImplementedError\n\ndef sample_top_p(logits, top_p, generator=None):\n    return torch.multinomial(top_p_probs(logits, top_p), 1, generator=generator).item()\n\nlogits = torch.log(torch.tensor([0.15, 0.60, 0.25], dtype=torch.float64))\nprobs = top_p_probs(logits, 0.7)\nexpected = torch.tensor([0., 0.60 / 0.85, 0.25 / 0.85], dtype=torch.float64)\nassert torch.allclose(probs, expected, atol=1e-12)\nassert torch.allclose(probs.sum(), torch.tensor(1., dtype=torch.float64))\nassert torch.equal(top_p_probs(logits, 0.3), torch.tensor([0., 1., 0.], dtype=torch.float64))\nassert torch.allclose(top_p_probs(logits, 1.0), torch.tensor([0.15, 0.60, 0.25], dtype=torch.float64))\ngenerator = torch.Generator().manual_seed(9)\nsamples = [sample_top_p(logits, 0.7, generator) for _ in range(20)]\nassert set(samples) <= {1, 2}\nprint(probs)\n",
    "solutionCode": "import torch\n\ndef top_p_probs(logits, top_p):\n    if not 0 < top_p <= 1:\n        raise ValueError('top_p must be in (0, 1]')\n    probabilities = torch.softmax(logits, dim=-1)\n    sorted_probs, token_ids = probabilities.sort(descending=True)\n    cumulative = sorted_probs.cumsum(dim=-1)\n    keep = torch.cat((torch.ones(1, dtype=torch.bool, device=logits.device),\n                      cumulative[:-1] < top_p))\n    result = torch.zeros_like(probabilities)\n    result[token_ids] = sorted_probs.masked_fill(~keep, 0)\n    return result / result.sum()\n\ndef sample_top_p(logits, top_p, generator=None):\n    return torch.multinomial(top_p_probs(logits, top_p), 1, generator=generator).item()\n\nlogits = torch.log(torch.tensor([0.15, 0.60, 0.25], dtype=torch.float64))\nprobs = top_p_probs(logits, 0.7)\nexpected = torch.tensor([0., 0.60 / 0.85, 0.25 / 0.85], dtype=torch.float64)\nassert torch.allclose(probs, expected, atol=1e-12)\nassert torch.allclose(probs.sum(), torch.tensor(1., dtype=torch.float64))\nassert torch.equal(top_p_probs(logits, 0.3), torch.tensor([0., 1., 0.], dtype=torch.float64))\nassert torch.allclose(top_p_probs(logits, 1.0), torch.tensor([0.15, 0.60, 0.25], dtype=torch.float64))\ngenerator = torch.Generator().manual_seed(9)\nsamples = [sample_top_p(logits, 0.7, generator) for _ in range(20)]\nassert set(samples) <= {1, 2}\nprint(probs)\n",
    "checks": [
      "For top_p=0.7, retain probabilities 0.60 and 0.25; the second token crosses the threshold.",
      "The original token with probability 0.15 receives zero probability.",
      "Small top_p keeps the most likely token; top_p=1 preserves the full distribution."
    ]
  },
  {
    "id": "q50",
    "type": "coding",
    "topic": "LLM inference",
    "title": "Match cached decoding to full causal attention",
    "prompt": "Implement one attention step that appends a new key/value pair to an optional KV cache, attends with the current query, and returns (output, updated_cache). Each new tensor has shape (1, head_dim); the cache holds earlier keys and values. Compare token-by-token results with full causal attention over the same sequence. Queries and keys are already projected and positioned.",
    "answer": [
      "Append the new key and value to the history, then score the current query against all cached keys and take the weighted sum of cached values. The cache includes only past and current tokens, so this single-step calculation needs no additional causal mask. Earlier keys and values are reused, but the current query still reads the growing history."
    ],
    "takeaway": "A correct KV cache reuses past keys and values while preserving full causal-attention outputs.",
    "starterCode": "import math\nimport torch\n\ndef cached_attention_step(query, new_key, new_value, cache=None):\n    # TODO: implement the requested operation.\n    raise NotImplementedError\n\nq = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])\nk = torch.tensor([[1., 1.], [1., 0.], [0., 1.]])\nv = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])\nscores = (q @ k.T) / math.sqrt(q.shape[-1])\nfuture = torch.ones_like(scores, dtype=torch.bool).triu(1)\nreference = torch.softmax(scores.masked_fill(future, -torch.inf), dim=-1) @ v\ncache = None\noutputs = []\nfor i in range(q.shape[0]):\n    output, cache = cached_attention_step(q[i:i+1], k[i:i+1], v[i:i+1], cache)\n    outputs.append(output)\n    assert cache[0].shape == (i + 1, 2)\n    assert torch.equal(cache[0], k[:i+1])\n    assert torch.equal(cache[1], v[:i+1])\nassert torch.allclose(torch.cat(outputs), reference, atol=1e-6)\nassert torch.equal(outputs[0], v[:1])\nprint(torch.cat(outputs))\n",
    "solutionCode": "import math\nimport torch\n\ndef cached_attention_step(query, new_key, new_value, cache=None):\n    if cache is None:\n        keys, values = new_key, new_value\n    else:\n        keys = torch.cat((cache[0], new_key), dim=0)\n        values = torch.cat((cache[1], new_value), dim=0)\n    scores = (query @ keys.T) / math.sqrt(query.shape[-1])\n    output = torch.softmax(scores, dim=-1) @ values\n    return output, (keys, values)\n\nq = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])\nk = torch.tensor([[1., 1.], [1., 0.], [0., 1.]])\nv = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])\nscores = (q @ k.T) / math.sqrt(q.shape[-1])\nfuture = torch.ones_like(scores, dtype=torch.bool).triu(1)\nreference = torch.softmax(scores.masked_fill(future, -torch.inf), dim=-1) @ v\ncache = None\noutputs = []\nfor i in range(q.shape[0]):\n    output, cache = cached_attention_step(q[i:i+1], k[i:i+1], v[i:i+1], cache)\n    outputs.append(output)\n    assert cache[0].shape == (i + 1, 2)\n    assert torch.equal(cache[0], k[:i+1])\n    assert torch.equal(cache[1], v[:i+1])\nassert torch.allclose(torch.cat(outputs), reference, atol=1e-6)\nassert torch.equal(outputs[0], v[:1])\nprint(torch.cat(outputs))\n",
    "checks": [
      "The cache grows by one key/value row at each step.",
      "Cached tensors preserve the original keys and values in token order.",
      "Incremental outputs match full causal attention, including the first token’s output."
    ]
  },
  {
    "id": "q51",
    "type": "qa",
    "topic": "RL for LLMs",
    "title": "Tokens as actions",
    "prompt": "In token-level RL for an LLM, what are the state, action, policy, and episode?",
    "answer": [
      "The state is the prompt plus generated prefix. An action is the next token, sampled from the model policy. An episode ends at a stop token or length limit. Rewards may arrive per token or only after the completion; the chosen termination rule affects the learning problem."
    ],
    "takeaway": "Define the trajectory and termination rule before defining the reward."
  },
  {
    "id": "q52",
    "type": "qa",
    "topic": "RL for LLMs",
    "title": "Why subtract a baseline?",
    "prompt": "Why does REINFORCE often subtract a state-dependent baseline from returns?",
    "answer": [
      "A baseline independent of the sampled action has zero expected contribution when multiplied by the policy score gradient. Subtracting it can reduce variance without changing the expected policy gradient. An inaccurate baseline may fail to reduce variance; an action-dependent baseline needs a correction."
    ],
    "takeaway": "A state-only baseline preserves the expected policy gradient."
  },
  {
    "id": "q53",
    "type": "qa",
    "topic": "RL for LLMs",
    "title": "Reward versus advantage",
    "prompt": "How does an advantage differ from a raw reward?",
    "answer": [
      "A reward is feedback at a step. An advantage estimates how much better an action is than the policy’s expected outcome in that state, typically using returns or a critic. A positive reward can still produce negative advantage when the state baseline is higher."
    ],
    "takeaway": "Advantage compares an action to an expected outcome."
  },
  {
    "id": "q54",
    "type": "qa",
    "topic": "RL for LLMs",
    "title": "Terminal reward credit",
    "prompt": "How can a completion-level reward train earlier tokens?",
    "answer": [
      "Policy-gradient training weights the log probability gradient of sampled actions by a return or advantage. A terminal reward can therefore affect every preceding token. This credit assignment is noisy: a good completion does not establish that every token was helpful."
    ],
    "takeaway": "Sequence rewards provide noisy token-level credit."
  },
  {
    "id": "q55",
    "type": "qa",
    "topic": "RL for LLMs",
    "title": "PPO importance ratios",
    "prompt": "What does the ratio exp(log pi_new minus log pi_old) measure in PPO?",
    "answer": [
      "It compares the current policy probability of the sampled action with its probability under the rollout policy, conditioned on the same prefix. A ratio above one means that action became more likely. The denominator must come from the actual behavior policy used to collect the rollout."
    ],
    "takeaway": "Keep rollout log probabilities fixed during PPO updates."
  },
  {
    "id": "q56",
    "type": "qa",
    "topic": "RL for LLMs",
    "title": "What clipping guarantees",
    "prompt": "Does the PPO clipped objective guarantee a maximum KL divergence?",
    "answer": [
      "No. The minimum of clipped and unclipped advantage terms limits some incentives to change sampled action probabilities. It is not a hard trust region or a global KL bound. Monitor policy drift and use suitable update counts, learning rates, or KL-based stopping."
    ],
    "takeaway": "PPO clipping is a surrogate objective, not a hard KL constraint."
  },
  {
    "id": "q57",
    "type": "qa",
    "topic": "RL for LLMs",
    "title": "Reference versus rollout policy",
    "prompt": "Why might RL training keep both a reference policy and an old policy?",
    "answer": [
      "The old policy generated the rollouts and supplies importance-ratio denominators. A reference policy anchors behavior through a divergence penalty, often relative to an SFT checkpoint. They serve different roles and need not be identical after training progresses."
    ],
    "takeaway": "The rollout policy corrects sampling; the reference policy anchors behavior."
  },
  {
    "id": "q58",
    "type": "qa",
    "topic": "RL for LLMs",
    "title": "Reward hacking",
    "prompt": "A reward score improves while human judgments worsen. What should you investigate?",
    "answer": [
      "The policy may exploit weaknesses in the reward model or evaluator, such as verbosity or formatting shortcuts. Inspect high-scoring failures, use held-out human judgments and diverse tasks, and revise the reward or constraints. Optimizing a proxy can amplify its errors."
    ],
    "takeaway": "Evaluate the intended behavior independently of the optimized reward."
  },
  {
    "id": "q59",
    "type": "qa",
    "topic": "RL for LLMs",
    "title": "Group-relative advantages",
    "prompt": "Why normalize rewards across multiple responses to the same prompt?",
    "answer": [
      "Within-prompt comparisons can remove prompt difficulty effects and provide a baseline without a separate value model. They need several sampled responses and depend on reward variation. If every response receives the same reward, the centered group signal is zero; a small epsilon only prevents division by zero."
    ],
    "takeaway": "Group-relative learning needs useful within-prompt reward differences."
  },
  {
    "id": "q60",
    "type": "qa",
    "topic": "RL for LLMs",
    "title": "Sparse verifier rewards",
    "prompt": "What are strengths and limitations of using test outcomes as coding rewards?",
    "answer": [
      "Executable tests provide repeatable feedback for tested behavior and can reduce subjective grading. Incomplete tests miss errors, and insecure execution can be dangerous. Use isolated execution and held-out tests; passing visible tests does not prove general correctness."
    ],
    "takeaway": "A verifier is only as complete as its checks."
  },
  {
    "id": "q61",
    "type": "qa",
    "topic": "RL for LLMs",
    "title": "DPO data assumptions",
    "prompt": "What does DPO learn from a chosen and rejected completion pair?",
    "answer": [
      "DPO increases the relative preference margin under the policy compared with a reference policy using a logistic objective. It uses offline pairs rather than collecting reward-model rollouts in the basic setup. Label noise, response style confounds, and coverage of the preference dataset still matter."
    ],
    "takeaway": "Preference pairs encode a relative training signal with dataset limitations."
  },
  {
    "id": "q62",
    "type": "qa",
    "topic": "RL for LLMs",
    "title": "KL direction matters",
    "prompt": "Why distinguish KL(policy || reference) from KL(reference || policy)?",
    "answer": [
      "KL is asymmetric. The first averages log probability ratios under policy samples; the second averages under reference samples. They weight regions differently and require different sampling or estimators. State the direction and reduction before comparing reported penalties."
    ],
    "takeaway": "Name the sampling distribution when reporting KL."
  },
  {
    "id": "q63",
    "type": "qa",
    "topic": "RL for LLMs",
    "title": "Truncation and termination",
    "prompt": "Why distinguish a true terminal state from a rollout cut off by a time limit?",
    "answer": [
      "At a true terminal state there is no future return, so a critic bootstrap is usually zero. At an artificial collection boundary, the underlying process may continue, so bootstrapping can be appropriate. For capped text generation, explicitly define whether reaching the cap is terminal in the task."
    ],
    "takeaway": "Bootstrap according to task semantics, not just a batch boundary."
  },
  {
    "id": "q64",
    "type": "qa",
    "topic": "RL for LLMs",
    "title": "On-policy freshness",
    "prompt": "Why can reusing old rollouts for many updates destabilize policy optimization?",
    "answer": [
      "As the policy changes, the rollout distribution becomes stale. Importance weighting can correct expectations under suitable support assumptions, but large ratios increase variance and clipping introduces bias. Limit reuse and monitor ratios and KL rather than assuming old samples remain representative."
    ],
    "takeaway": "More reuse trades sampling efficiency against distribution drift."
  },
  {
    "id": "q65",
    "type": "qa",
    "topic": "RL for LLMs",
    "title": "Length bias in rewards",
    "prompt": "How can a reward pipeline accidentally favor longer responses?",
    "answer": [
      "Summed token rewards or log-probability terms scale with length, and a learned judge may prefer verbosity. Inspect reward versus length, control comparisons for length, and choose sequence or token reductions intentionally. Length normalization changes the objective and is not always the right fix."
    ],
    "takeaway": "Audit length effects before interpreting reward gains."
  },
  {
    "id": "q66",
    "type": "qa",
    "topic": "LLM fundamentals",
    "title": "Tokenizer boundaries",
    "prompt": "Why can concatenating separately tokenized strings differ from tokenizing their concatenation?",
    "answer": [
      "Subword tokenization depends on surrounding bytes or characters, including whitespace. A boundary can merge or split pieces differently. Build the complete formatted conversation before tokenizing when the template expects it, and verify special token handling."
    ],
    "takeaway": "Token boundaries depend on context and formatting."
  },
  {
    "id": "q67",
    "type": "qa",
    "topic": "LLM fundamentals",
    "title": "Perplexity comparisons",
    "prompt": "When is comparing perplexity across two LLMs misleading?",
    "answer": [
      "Perplexity is exp(mean token negative log likelihood), so it depends on tokenizer, corpus, preprocessing, and which tokens are scored. Different token units make raw values hard to compare. Use a common evaluation protocol and consider byte-normalized likelihood where appropriate."
    ],
    "takeaway": "Perplexity needs matched tokenization and scoring conditions."
  },
  {
    "id": "q68",
    "type": "qa",
    "topic": "LLM fundamentals",
    "title": "Exposure bias",
    "prompt": "What mismatch exists between teacher forcing and autoregressive generation?",
    "answer": [
      "Teacher forcing conditions on gold prefixes; generation conditions on the model’s sampled prefixes. Mistakes can create prefixes rarely seen in training and compound downstream. Evaluate generated sequences in addition to teacher-forced loss."
    ],
    "takeaway": "Low teacher-forced loss does not fully characterize generation."
  },
  {
    "id": "q69",
    "type": "qa",
    "topic": "LLM fundamentals",
    "title": "Position extrapolation",
    "prompt": "Why does accepting a longer input not prove reliable long-context reasoning?",
    "answer": [
      "The implementation may support a larger position range while attention patterns or positional representations generalize poorly beyond training lengths. Evaluate retrieval and reasoning at multiple positions and lengths, including distractors and evidence near the middle."
    ],
    "takeaway": "Context capacity and effective context use are separate measurements."
  },
  {
    "id": "q70",
    "type": "qa",
    "topic": "LLM fundamentals",
    "title": "RAG chunk boundaries",
    "prompt": "How do chunk size and overlap affect retrieval-augmented generation?",
    "answer": [
      "Small chunks can improve precision but lose context; large chunks preserve context but add irrelevant material and consume the context budget. Overlap can preserve boundary information but duplicates evidence. Tune using retrieval relevance and final answer quality on representative questions."
    ],
    "takeaway": "Tune retrieval chunks against downstream answers."
  },
  {
    "id": "q71",
    "type": "qa",
    "topic": "LLM fundamentals",
    "title": "Retrieval failure or generation failure",
    "prompt": "How would you separate retrieval errors from answer-generation errors in RAG?",
    "answer": [
      "Check whether the retrieved passages contain sufficient correct evidence. Compare answers with retrieved context against answers with an oracle evidence set. If oracle context helps, retrieval may be limiting; if it does not, inspect prompting, reasoning, ambiguity, and evaluation."
    ],
    "takeaway": "Evaluate evidence availability separately from evidence use."
  },
  {
    "id": "q72",
    "type": "qa",
    "topic": "LLM fundamentals",
    "title": "Prompt injection in retrieved text",
    "prompt": "Why should retrieved documents not be treated as trusted instructions?",
    "answer": [
      "External documents can contain adversarial instructions that conflict with the user’s task. Delimit evidence, enforce instruction priority, restrict tool permissions, and validate consequential outputs. Delimiters alone do not guarantee resistance."
    ],
    "takeaway": "Treat retrieved text as untrusted evidence."
  },
  {
    "id": "q73",
    "type": "qa",
    "topic": "LLM fundamentals",
    "title": "Structured output validation",
    "prompt": "Does syntactically valid JSON imply a correct tool argument?",
    "answer": [
      "No. A schema can enforce types and structure but cannot establish that an identifier exists or an action is appropriate. Validate semantic constraints and authorization in application code, and handle parsing or validation failures explicitly."
    ],
    "takeaway": "Validate meaning and permissions after structure."
  },
  {
    "id": "q74",
    "type": "qa",
    "topic": "LLM fundamentals",
    "title": "Sampling temperature",
    "prompt": "How does positive temperature affect token sampling?",
    "answer": [
      "Divide logits by temperature before softmax. Lower positive values concentrate mass on higher logits; higher values flatten the distribution. Temperature zero is normally implemented as a separate greedy choice, not literal division by zero."
    ],
    "takeaway": "Temperature changes the distribution; greedy decoding is a separate case."
  },
  {
    "id": "q75",
    "type": "qa",
    "topic": "LLM fundamentals",
    "title": "Top-p ordering",
    "prompt": "Why can applying top-p before temperature differ from applying it afterward?",
    "answer": [
      "Top-p retains the smallest sorted prefix whose cumulative probability reaches a threshold. Temperature changes those probabilities and can change the retained set. Specify the operation order when reproducing decoding behavior."
    ],
    "takeaway": "Sampling transforms generally do not commute."
  },
  {
    "id": "q76",
    "type": "qa",
    "topic": "LLM fundamentals",
    "title": "Quantization calibration",
    "prompt": "Why should quantization calibration data resemble deployment inputs?",
    "answer": [
      "Calibration estimates activation ranges or related statistics. Unrepresentative inputs can miss outliers and cause clipping or poor scaling at inference. Evaluate accuracy across target tasks and sequence lengths; weight-only quantization has different calibration needs from activation quantization."
    ],
    "takeaway": "Calibrate and evaluate for the actual quantization scheme and workload."
  },
  {
    "id": "q77",
    "type": "qa",
    "topic": "LLM fundamentals",
    "title": "Distillation targets",
    "prompt": "What information can soft teacher distributions provide beyond hard labels?",
    "answer": [
      "They express relative probabilities across alternatives, including less likely tokens. Distillation can match these distributions, often with a temperature, but teacher errors and domain gaps remain. Matching only sampled teacher outputs provides a different signal from full-distribution distillation."
    ],
    "takeaway": "Specify whether distillation uses distributions or generated text."
  },
  {
    "id": "q78",
    "type": "qa",
    "topic": "LLM fundamentals",
    "title": "Benchmark contamination",
    "prompt": "Why can benchmark overlap inflate measured LLM performance?",
    "answer": [
      "Training data may contain evaluation questions, solutions, or close paraphrases, allowing memorization to substitute for generalization. Audit data provenance and overlap, use fresh held-out tasks, and report uncertainty when training data is unavailable."
    ],
    "takeaway": "A held-out file is not necessarily held out from pretraining."
  },
  {
    "id": "q79",
    "type": "qa",
    "topic": "LLM fundamentals",
    "title": "Tool call evaluation",
    "prompt": "What should an evaluation check beyond whether an LLM selected the correct tool name?",
    "answer": [
      "Check argument correctness, authorization, handling of tool failures, unnecessary calls, and whether the final response accurately reflects results. Evaluate multi-step outcomes in a controlled environment because a correct first call can still lead to failure."
    ],
    "takeaway": "Evaluate complete tool-assisted outcomes."
  },
  {
    "id": "q80",
    "type": "qa",
    "topic": "LLM fundamentals",
    "title": "Abstention and calibration",
    "prompt": "Why is a fluent explanation not a reliable confidence score?",
    "answer": [
      "Fluency is learned language behavior and does not establish correctness. Evaluate confidence or abstention decisions against labeled outcomes, including distribution shifts. Use evidence checks and explicit uncertainty policies instead of treating persuasive wording as calibrated probability."
    ],
    "takeaway": "Measure confidence against correctness, not eloquence."
  },
  {
    "id": "q81",
    "type": "coding",
    "topic": "LLM coding",
    "title": "Stable log-softmax",
    "prompt": "Return log probabilities for a nonempty list of finite logits without exponentiating large raw logits.",
    "answer": [
      "Shift logits before exponentiation. The reference implementation below follows the stated input contract."
    ],
    "takeaway": "Shift logits before exponentiation.",
    "starterCode": "import math\n\ndef log_softmax(xs):\n    # TODO: implement the operation described above.\n    raise NotImplementedError\n\nassert all(abs(x+math.log(2))<1e-10 for x in log_softmax([1000.,1000.]))\n",
    "solutionCode": "import math\n\ndef log_softmax(xs):\n    m=max(xs)\n    z=m+math.log(sum(math.exp(x-m) for x in xs))\n    return [x-z for x in xs]\n\nassert all(abs(x+math.log(2))<1e-10 for x in log_softmax([1000.,1000.]))\n",
    "checks": [
      "Run the supplied assertions, then add a case with different inputs.",
      "Explain the input contract and the normalization or boundary rule used."
    ]
  },
  {
    "id": "q82",
    "type": "coding",
    "topic": "LLM coding",
    "title": "Temperature distribution",
    "prompt": "Return softmax probabilities; reject nonpositive temperature with ValueError.",
    "answer": [
      "Validate temperature before scaling. The reference implementation below follows the stated input contract."
    ],
    "takeaway": "Validate temperature before scaling.",
    "starterCode": "import math\n\ndef temperature_probs(xs, temperature):\n    # TODO: implement the operation described above.\n    raise NotImplementedError\n\nassert temperature_probs([2.,2.],1.) == [0.5,0.5]\ntry: temperature_probs([1.],0.)\nexcept ValueError: pass\nelse: raise AssertionError(\"must reject zero\")\n",
    "solutionCode": "import math\n\ndef temperature_probs(xs, temperature):\n    if temperature<=0: raise ValueError(\"positive temperature required\")\n    scaled=[x/temperature for x in xs]\n    m=max(scaled)\n    weights=[math.exp(x-m) for x in scaled]\n    return [x/sum(weights) for x in weights]\n\nassert temperature_probs([2.,2.],1.) == [0.5,0.5]\ntry: temperature_probs([1.],0.)\nexcept ValueError: pass\nelse: raise AssertionError(\"must reject zero\")\n",
    "checks": [
      "Run the supplied assertions, then add a case with different inputs.",
      "Explain the input contract and the normalization or boundary rule used."
    ]
  },
  {
    "id": "q83",
    "type": "coding",
    "topic": "LLM coding",
    "title": "Nucleus candidate set",
    "prompt": "For normalized nonnegative probabilities and 0 < p <= 1, return indices in descending probability order through the first cumulative mass >= p. Keep at least one; break ties by input order.",
    "answer": [
      "Include the token that crosses the mass threshold. The reference implementation below follows the stated input contract."
    ],
    "takeaway": "Include the token that crosses the mass threshold.",
    "starterCode": "import math\n\ndef nucleus(probs, p):\n    # TODO: implement the operation described above.\n    raise NotImplementedError\n\nassert nucleus([.1,.6,.3],.8)==[1,2]\nassert nucleus([.7,.3],.1)==[0]\n",
    "solutionCode": "import math\n\ndef nucleus(probs, p):\n    if not 0<p<=1: raise ValueError(\"invalid p\")\n    result=[]\n    total=0.\n    for i in sorted(range(len(probs)), key=lambda i:-probs[i]):\n        result.append(i)\n        total+=probs[i]\n        if total>=p: break\n    return result\n\nassert nucleus([.1,.6,.3],.8)==[1,2]\nassert nucleus([.7,.3],.1)==[0]\n",
    "checks": [
      "Run the supplied assertions, then add a case with different inputs.",
      "Explain the input contract and the normalization or boundary rule used."
    ]
  },
  {
    "id": "q84",
    "type": "coding",
    "topic": "LLM coding",
    "title": "Shift next-token labels",
    "prompt": "Given a sequence with at least two tokens, return input and next-token target lists.",
    "answer": [
      "Align every input position with its next token. The reference implementation below follows the stated input contract."
    ],
    "takeaway": "Align every input position with its next token.",
    "starterCode": "import math\n\ndef shift_tokens(ids):\n    # TODO: implement the operation described above.\n    raise NotImplementedError\n\nassert shift_tokens([1,7,8,2])==([1,7,8],[7,8,2])\n",
    "solutionCode": "import math\n\ndef shift_tokens(ids):\n    return ids[:-1],ids[1:]\n\nassert shift_tokens([1,7,8,2])==([1,7,8],[7,8,2])\n",
    "checks": [
      "Run the supplied assertions, then add a case with different inputs.",
      "Explain the input contract and the normalization or boundary rule used."
    ]
  },
  {
    "id": "q85",
    "type": "coding",
    "topic": "LLM coding",
    "title": "Assistant loss mask",
    "prompt": "Average losses where the same-length binary mask is 1. Return 0.0 when no positions are selected.",
    "answer": [
      "Normalize by supervised tokens. The reference implementation below follows the stated input contract."
    ],
    "takeaway": "Normalize by supervised tokens.",
    "starterCode": "import math\n\ndef masked_mean(losses, mask):\n    # TODO: implement the operation described above.\n    raise NotImplementedError\n\nassert masked_mean([9.,2.,4.],[0,1,1])==3.\nassert masked_mean([9.],[0])==0.\n",
    "solutionCode": "import math\n\ndef masked_mean(losses, mask):\n    n=sum(mask)\n    return sum(x*m for x,m in zip(losses,mask))/n if n else 0.0\n\nassert masked_mean([9.,2.,4.],[0,1,1])==3.\nassert masked_mean([9.],[0])==0.\n",
    "checks": [
      "Run the supplied assertions, then add a case with different inputs.",
      "Explain the input contract and the normalization or boundary rule used."
    ]
  },
  {
    "id": "q86",
    "type": "coding",
    "topic": "LLM coding",
    "title": "Causal visibility",
    "prompt": "Return an n by n Boolean list matrix with True for visible keys, including the diagonal.",
    "answer": [
      "Visibility includes the current position. The reference implementation below follows the stated input contract."
    ],
    "takeaway": "Visibility includes the current position.",
    "starterCode": "import math\n\ndef causal_mask(n):\n    # TODO: implement the operation described above.\n    raise NotImplementedError\n\nassert causal_mask(3)==[[True,False,False],[True,True,False],[True,True,True]]\n",
    "solutionCode": "import math\n\ndef causal_mask(n):\n    return [[j<=i for j in range(n)] for i in range(n)]\n\nassert causal_mask(3)==[[True,False,False],[True,True,False],[True,True,True]]\n",
    "checks": [
      "Run the supplied assertions, then add a case with different inputs.",
      "Explain the input contract and the normalization or boundary rule used."
    ]
  },
  {
    "id": "q87",
    "type": "coding",
    "topic": "LLM coding",
    "title": "Pad a token batch",
    "prompt": "For a nonempty batch of token lists, right-pad to the longest length and return padded rows plus 1/0 attention masks. Empty rows are allowed.",
    "answer": [
      "Build masks from lengths, not token values. The reference implementation below follows the stated input contract."
    ],
    "takeaway": "Build masks from lengths, not token values.",
    "starterCode": "import math\n\ndef pad_batch(batch, pad_id):\n    # TODO: implement the operation described above.\n    raise NotImplementedError\n\nassert pad_batch([[3,4],[]],0)==([[3,4],[0,0]],[[1,1],[0,0]])\n",
    "solutionCode": "import math\n\ndef pad_batch(batch, pad_id):\n    width=max(map(len,batch))\n    return ([row+[pad_id]*(width-len(row)) for row in batch], [[1]*len(row)+[0]*(width-len(row)) for row in batch])\n\nassert pad_batch([[3,4],[]],0)==([[3,4],[0,0]],[[1,1],[0,0]])\n",
    "checks": [
      "Run the supplied assertions, then add a case with different inputs.",
      "Explain the input contract and the normalization or boundary rule used."
    ]
  },
  {
    "id": "q88",
    "type": "coding",
    "topic": "LLM coding",
    "title": "Prompt-free completion score",
    "prompt": "Sum already aligned token log probabilities after prompt_length prompt tokens. Do not shift these values again.",
    "answer": [
      "Score only the intended completion positions. The reference implementation below follows the stated input contract."
    ],
    "takeaway": "Score only the intended completion positions.",
    "starterCode": "import math\n\ndef completion_score(logps, prompt_length):\n    # TODO: implement the operation described above.\n    raise NotImplementedError\n\nassert abs(completion_score([-9.,-8.,-.2,-.3],2)+.5)<1e-10\n",
    "solutionCode": "import math\n\ndef completion_score(logps, prompt_length):\n    return sum(logps[prompt_length:])\n\nassert abs(completion_score([-9.,-8.,-.2,-.3],2)+.5)<1e-10\n",
    "checks": [
      "Run the supplied assertions, then add a case with different inputs.",
      "Explain the input contract and the normalization or boundary rule used."
    ]
  },
  {
    "id": "q89",
    "type": "coding",
    "topic": "LLM coding",
    "title": "Perplexity from token losses",
    "prompt": "Return exp(mean negative log likelihood) for a nonempty token-loss list.",
    "answer": [
      "Exponentiate the mean token loss. The reference implementation below follows the stated input contract."
    ],
    "takeaway": "Exponentiate the mean token loss.",
    "starterCode": "import math\n\ndef perplexity(nlls):\n    # TODO: implement the operation described above.\n    raise NotImplementedError\n\nassert abs(perplexity([math.log(2),math.log(2)])-2)<1e-10\n",
    "solutionCode": "import math\n\ndef perplexity(nlls):\n    return math.exp(sum(nlls)/len(nlls))\n\nassert abs(perplexity([math.log(2),math.log(2)])-2)<1e-10\n",
    "checks": [
      "Run the supplied assertions, then add a case with different inputs.",
      "Explain the input contract and the normalization or boundary rule used."
    ]
  },
  {
    "id": "q90",
    "type": "coding",
    "topic": "LLM coding",
    "title": "Greedy repetition penalty",
    "prompt": "Return adjusted logits for penalty >= 1: divide positive seen-token logits by penalty and multiply negative ones. Apply once per unique seen index.",
    "answer": [
      "Handle signed logits and duplicate history tokens. The reference implementation below follows the stated input contract."
    ],
    "takeaway": "Handle signed logits and duplicate history tokens.",
    "starterCode": "import math\n\ndef penalize(logits, seen, penalty):\n    # TODO: implement the operation described above.\n    raise NotImplementedError\n\nassert penalize([4.,-2.,1.],[0,0,1],2.)==[2.,-4.,1.]\n",
    "solutionCode": "import math\n\ndef penalize(logits, seen, penalty):\n    out=list(logits)\n    for i in set(seen):\n        out[i]=out[i]/penalty if out[i]>0 else out[i]*penalty\n    return out\n\nassert penalize([4.,-2.,1.],[0,0,1],2.)==[2.,-4.,1.]\n",
    "checks": [
      "Run the supplied assertions, then add a case with different inputs.",
      "Explain the input contract and the normalization or boundary rule used."
    ]
  },
  {
    "id": "q91",
    "type": "coding",
    "topic": "LLM coding",
    "title": "KV cache accounting",
    "prompt": "Compute K plus V storage bytes, excluding allocator overhead and metadata.",
    "answer": [
      "Count both keys and values using KV heads. The reference implementation below follows the stated input contract."
    ],
    "takeaway": "Count both keys and values using KV heads.",
    "starterCode": "import math\n\ndef kv_bytes(layers, kv_heads, head_dim, tokens, batch, bytes_per_element):\n    # TODO: implement the operation described above.\n    raise NotImplementedError\n\nassert kv_bytes(2,4,8,10,3,2)==7680\n",
    "solutionCode": "import math\n\ndef kv_bytes(layers, kv_heads, head_dim, tokens, batch, bytes_per_element):\n    return 2*layers*kv_heads*head_dim*tokens*batch*bytes_per_element\n\nassert kv_bytes(2,4,8,10,3,2)==7680\n",
    "checks": [
      "Run the supplied assertions, then add a case with different inputs.",
      "Explain the input contract and the normalization or boundary rule used."
    ]
  },
  {
    "id": "q92",
    "type": "coding",
    "topic": "LLM coding",
    "title": "Chunk retrieval text",
    "prompt": "Return overlapping token chunks. Require size > 0 and 0 <= overlap < size; avoid a trailing chunk entirely contained in the previous chunk. Empty input gives [].",
    "answer": [
      "Stop after a chunk reaches the end. The reference implementation below follows the stated input contract."
    ],
    "takeaway": "Stop after a chunk reaches the end.",
    "starterCode": "import math\n\ndef chunks(tokens, size, overlap):\n    # TODO: implement the operation described above.\n    raise NotImplementedError\n\nassert chunks(list(range(6)),4,2)==[[0,1,2,3],[2,3,4,5]]\nassert chunks([],4,1)==[]\n",
    "solutionCode": "import math\n\ndef chunks(tokens, size, overlap):\n    if size<=0 or not 0<=overlap<size: raise ValueError(\"invalid chunk settings\")\n    out=[]\n    start=0\n    while start<len(tokens):\n        out.append(tokens[start:start+size])\n        if start+size>=len(tokens): break\n        start+=size-overlap\n    return out\n\nassert chunks(list(range(6)),4,2)==[[0,1,2,3],[2,3,4,5]]\nassert chunks([],4,1)==[]\n",
    "checks": [
      "Run the supplied assertions, then add a case with different inputs.",
      "Explain the input contract and the normalization or boundary rule used."
    ]
  },
  {
    "id": "q93",
    "type": "coding",
    "topic": "LLM coding",
    "title": "Cosine retrieval score",
    "prompt": "Return cosine similarity of equal-length vectors. Define the score as 0.0 if either vector has zero norm.",
    "answer": [
      "Handle zero vectors explicitly. The reference implementation below follows the stated input contract."
    ],
    "takeaway": "Handle zero vectors explicitly.",
    "starterCode": "import math\n\ndef cosine(a, b):\n    # TODO: implement the operation described above.\n    raise NotImplementedError\n\nassert abs(cosine([1,0],[1,1])-1/math.sqrt(2))<1e-10\nassert cosine([0,0],[1,2])==0.\n",
    "solutionCode": "import math\n\ndef cosine(a, b):\n    den=math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))\n    return sum(x*y for x,y in zip(a,b))/den if den else 0.0\n\nassert abs(cosine([1,0],[1,1])-1/math.sqrt(2))<1e-10\nassert cosine([0,0],[1,2])==0.\n",
    "checks": [
      "Run the supplied assertions, then add a case with different inputs.",
      "Explain the input contract and the normalization or boundary rule used."
    ]
  },
  {
    "id": "q94",
    "type": "coding",
    "topic": "LLM coding",
    "title": "Reciprocal rank",
    "prompt": "Return reciprocal rank of the first relevant document ID, or zero when none is retrieved. relevant is a set.",
    "answer": [
      "Ranks start at one. The reference implementation below follows the stated input contract."
    ],
    "takeaway": "Ranks start at one.",
    "starterCode": "import math\n\ndef reciprocal_rank(ranked, relevant):\n    # TODO: implement the operation described above.\n    raise NotImplementedError\n\nassert reciprocal_rank([\"a\",\"b\",\"c\"],{\"b\",\"c\"})==.5\nassert reciprocal_rank([\"a\"],{\"z\"})==0.\n",
    "solutionCode": "import math\n\ndef reciprocal_rank(ranked, relevant):\n    return next((1/i for i,doc in enumerate(ranked,1) if doc in relevant),0.0)\n\nassert reciprocal_rank([\"a\",\"b\",\"c\"],{\"b\",\"c\"})==.5\nassert reciprocal_rank([\"a\"],{\"z\"})==0.\n",
    "checks": [
      "Run the supplied assertions, then add a case with different inputs.",
      "Explain the input contract and the normalization or boundary rule used."
    ]
  },
  {
    "id": "q95",
    "type": "coding",
    "topic": "LLM coding",
    "title": "Discounted sequence returns",
    "prompt": "Compute discounted reward-to-go at every step for 0 <= gamma <= 1. Empty input gives [].",
    "answer": [
      "Accumulate returns backward. The reference implementation below follows the stated input contract."
    ],
    "takeaway": "Accumulate returns backward.",
    "starterCode": "import math\n\ndef returns(rewards, gamma):\n    # TODO: implement the operation described above.\n    raise NotImplementedError\n\nassert returns([0.,0.,1.],.5)==[.25,.5,1.]\nassert returns([],1.)==[]\n",
    "solutionCode": "import math\n\ndef returns(rewards, gamma):\n    out=[0.]*len(rewards)\n    value=0.\n    for i in range(len(rewards)-1,-1,-1):\n        value=rewards[i]+gamma*value\n        out[i]=value\n    return out\n\nassert returns([0.,0.,1.],.5)==[.25,.5,1.]\nassert returns([],1.)==[]\n",
    "checks": [
      "Run the supplied assertions, then add a case with different inputs.",
      "Explain the input contract and the normalization or boundary rule used."
    ]
  },
  {
    "id": "q96",
    "type": "coding",
    "topic": "LLM coding",
    "title": "Group-relative rewards",
    "prompt": "Center a nonempty reward group and divide by population standard deviation plus eps. Equal rewards must yield zeros.",
    "answer": [
      "Center within each prompt group. The reference implementation below follows the stated input contract."
    ],
    "takeaway": "Center within each prompt group.",
    "starterCode": "import math\n\ndef group_advantages(rewards, eps=1e-8):\n    # TODO: implement the operation described above.\n    raise NotImplementedError\n\nassert group_advantages([3.,3.])==[0.,0.]\na=group_advantages([1.,3.]); assert a[0]<0<a[1] and abs(sum(a))<1e-10\n",
    "solutionCode": "import math\n\ndef group_advantages(rewards, eps=1e-8):\n    mean=sum(rewards)/len(rewards)\n    sd=math.sqrt(sum((r-mean)**2 for r in rewards)/len(rewards))\n    return [(r-mean)/(sd+eps) for r in rewards]\n\nassert group_advantages([3.,3.])==[0.,0.]\na=group_advantages([1.,3.]); assert a[0]<0<a[1] and abs(sum(a))<1e-10\n",
    "checks": [
      "Run the supplied assertions, then add a case with different inputs.",
      "Explain the input contract and the normalization or boundary rule used."
    ]
  },
  {
    "id": "q97",
    "type": "coding",
    "topic": "LLM coding",
    "title": "PPO clipped objective",
    "prompt": "Return the scalar objective to maximize: min(ratio*A, clipped_ratio*A). Inputs are finite and epsilon is in (0,1).",
    "answer": [
      "The sign of advantage matters when taking the minimum. The reference implementation below follows the stated input contract."
    ],
    "takeaway": "The sign of advantage matters when taking the minimum.",
    "starterCode": "import math\n\ndef ppo_term(new_logp, old_logp, advantage, epsilon):\n    # TODO: implement the operation described above.\n    raise NotImplementedError\n\nassert abs(ppo_term(math.log(1.5),0.,2.,.2)-2.4)<1e-10\nassert abs(ppo_term(math.log(1.5),0.,-2.,.2)+3.)<1e-10\n",
    "solutionCode": "import math\n\ndef ppo_term(new_logp, old_logp, advantage, epsilon):\n    ratio=math.exp(new_logp-old_logp)\n    clipped=max(1-epsilon,min(1+epsilon,ratio))\n    return min(ratio*advantage,clipped*advantage)\n\nassert abs(ppo_term(math.log(1.5),0.,2.,.2)-2.4)<1e-10\nassert abs(ppo_term(math.log(1.5),0.,-2.,.2)+3.)<1e-10\n",
    "checks": [
      "Run the supplied assertions, then add a case with different inputs.",
      "Explain the input contract and the normalization or boundary rule used."
    ]
  },
  {
    "id": "q98",
    "type": "coding",
    "topic": "LLM coding",
    "title": "DPO pair loss",
    "prompt": "Given sequence log-probability sums and positive beta, compute -log sigmoid(beta*((chosen-rejected)-(ref_chosen-ref_rejected))) stably.",
    "answer": [
      "Subtract the reference margin and use stable softplus. The reference implementation below follows the stated input contract."
    ],
    "takeaway": "Subtract the reference margin and use stable softplus.",
    "starterCode": "import math\n\ndef dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta):\n    # TODO: implement the operation described above.\n    raise NotImplementedError\n\nassert abs(dpo_loss(-2.,-3.,-2.,-3.,.1)-math.log(2))<1e-10\nassert math.isfinite(dpo_loss(-2000.,0.,0.,0.,1.))\n",
    "solutionCode": "import math\n\ndef dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta):\n    z=beta*((chosen-rejected)-(ref_chosen-ref_rejected))\n    return max(0.,-z)+math.log1p(math.exp(-abs(z)))\n\nassert abs(dpo_loss(-2.,-3.,-2.,-3.,.1)-math.log(2))<1e-10\nassert math.isfinite(dpo_loss(-2000.,0.,0.,0.,1.))\n",
    "checks": [
      "Run the supplied assertions, then add a case with different inputs.",
      "Explain the input contract and the normalization or boundary rule used."
    ]
  },
  {
    "id": "q99",
    "type": "coding",
    "topic": "LLM coding",
    "title": "Exact categorical KL",
    "prompt": "For normalized distributions with q strictly positive, compute KL(p || q). Treat zero-p terms as zero.",
    "answer": [
      "Weight log ratios by the first distribution. The reference implementation below follows the stated input contract."
    ],
    "takeaway": "Weight log ratios by the first distribution.",
    "starterCode": "import math\n\ndef categorical_kl(p, q):\n    # TODO: implement the operation described above.\n    raise NotImplementedError\n\nassert abs(categorical_kl([1.,0.],[.5,.5])-math.log(2))<1e-10\nassert categorical_kl([.5,.5],[.5,.5])==0.\n",
    "solutionCode": "import math\n\ndef categorical_kl(p, q):\n    return sum(x*math.log(x/y) for x,y in zip(p,q) if x>0)\n\nassert abs(categorical_kl([1.,0.],[.5,.5])-math.log(2))<1e-10\nassert categorical_kl([.5,.5],[.5,.5])==0.\n",
    "checks": [
      "Run the supplied assertions, then add a case with different inputs.",
      "Explain the input contract and the normalization or boundary rule used."
    ]
  },
  {
    "id": "q100",
    "type": "coding",
    "topic": "LLM coding",
    "title": "Low-rank update dimensions",
    "prompt": "Return the number of trainable entries in A of shape (rank,in_features) and B of shape (out_features,rank), without bias.",
    "answer": [
      "Both low-rank factors contribute trainable parameters. The reference implementation below follows the stated input contract."
    ],
    "takeaway": "Both low-rank factors contribute trainable parameters.",
    "starterCode": "import math\n\ndef lora_parameters(in_features, out_features, rank):\n    # TODO: implement the operation described above.\n    raise NotImplementedError\n\nassert lora_parameters(100,200,4)==1200\n",
    "solutionCode": "import math\n\ndef lora_parameters(in_features, out_features, rank):\n    return rank*(in_features+out_features)\n\nassert lora_parameters(100,200,4)==1200\n",
    "checks": [
      "Run the supplied assertions, then add a case with different inputs.",
      "Explain the input contract and the normalization or boundary rule used."
    ]
  }
];
