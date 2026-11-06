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
  }
];
