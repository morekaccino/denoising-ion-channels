# KI-HMM: How We Taught a Neural Network the Physics of Ion Channels

*Illustrated edition* — plain-language companion to the novel method in this repository.

Front cover: AI illustration. Diagrams: generated from the repository's own results.

## The problem

# The problem

![A cell membrane with ion channels](membrane.png)

A cell membrane is full of ion channels: tiny gates that open and close. When one opens, a small current flows. In a patch-clamp experiment, we touch a cell with a glass pipette and record the current from the few channels inside the patch.

![A glass pipette touching a cell](patch.png)

Two things make the recording hard to read.

First, there is usually more than one channel in the patch. The amplifier records their **sum**. If two channels are open at once, the trace shows double the current of one open channel. We never see the channels individually.

Second, the recording is noisy. The current jumps up when channels open and down when they close, but noise blurs the steps.

![Current levels and noise](levels.png)

The figure shows what one channel contributes. A closed channel adds about 0.58 to the current. An open channel adds about 1.40. The step between them is about 0.82. With three channels, the possible clean levels are 1.74, 2.56, 3.38 and 4.20, depending on how many are open. The dots show what we actually record: the level you would expect, plus noise.

So from one squiggly line, we want two numbers:

- **N**: how many channels are in the patch.
- **k(t)**: how many of them are open at each moment, a number from 0 to N.

Why bother? CFTR is the channel that fails in cystic fibrosis. Drugs for the disease are judged by how they change opening and closing. To measure that, you first need to know how many channels you are looking at and what they are doing.

Everything in this book comes from a simulator where the truth is known, so every method can be scored against the answer key. Later, the same tools run on real recordings from the lab.

## The machine behind the curtain

# The machine behind the curtain

![A lamp behind frosted glass](lamps.png)

Imagine a lamp with two settings, dim and bright, behind frosted glass. You cannot see the lamp. You only see the glow, which is noisy. Watching the glow, you want to know when the lamp was bright.

This is a **hidden Markov model**, or HMM. Three pieces describe it:

1. **Hidden states.** What the lamp is doing. For channels: how many are open.
2. **Transitions.** The odds of switching states. The word **Markov** means the next state depends only on the current one, not on the whole past. A channel does not remember that it was open two seconds ago; only whether it is open now.
3. **Emissions.** What you observe in each state. A bright lamp glows strongly; an open channel raises the current.

For our channels, the transition rules come from the kinetic model: the measured rates at which a channel hops between its seven chemical states. The emission rule comes from the noise: each state predicts a current level, plus random wobble.

## Why the count vector is enough

There can be several channels, so a single state is not enough. Use a **count vector**:

```
n = (n1, n2, n3, n4, n5, n6, n7),   n1 + ... + n7 = N
```

Each `nj` counts how many channels sit in kinetic state `j`. For example, with 3 channels, `n = (2, 1, 0, 0, 0, 0, 0)` means 2 channels are in state 1 and 1 channel is in state 2.

![Why the count vector works](toy_markov.png)

The figure shows the idea with two channels. On the left, each channel has its own state, and because channels do not affect each other, the future of each one depends only on where it is now. That means the counts are all we need to predict what happens next: the count vector is a Markov chain.

On the right is a trap. Two different hidden setups can have the same open count. Here both have exactly one open channel, but the underlying states differ, and they have different odds of opening next. That is why the **open count alone** is not a perfect Markov chain: adding the counts throws away information. The exact decoder handles this by keeping all seven counts. The neural model later uses a careful approximation.

## The one-tick table

The kinetic model gives rates per second between the seven states. Collect them in a matrix `R`, then convert to a one-tick probability table with

```
P0 = expm(dt * R)
```

`dt` is the time between samples, 0.01 seconds. Why the exponential and not just multiply? Rates tell you the chance of hopping in a tiny instant: rate 10 per second, step 0.01 seconds, chance about 10 × 0.01 = 0.1. But over a full step, a channel could in principle hop twice, or three times. `expm` is the matrix version of the number `e^x`, and it sums all of those possibilities exactly. You will see a worked example in the math appendix.

## Counting the hidden possibilities

# Counting the hidden possibilities

Before we can compute anything, we need to know how many hidden setups exist. This chapter works out that number from scratch. It is the piece that looks scary as `C(N+6, 6)` and turns out to be simple.

## What C(a, b) means

`C(a, b)`, read "a choose b", counts the ways to pick `b` things out of `a` when order does not matter. The formula:

```
C(a, b) = a! / (b! * (a - b)!)
```

The exclamation mark means factorial: `4! = 4 × 3 × 2 × 1 = 24`. Example: how many two-card hands can you draw from five cards?

```
C(5, 2) = 5! / (2! * 3!) = 120 / (2 * 6) = 10
```

That is all the notation means.

## Why 6?

There are **7 kinetic states**, so the count vector has 7 slots: `n1` through `n7`. But they are not independent. They must add up to N. If you know the first six, the seventh is forced:

```
n7 = N - (n1 + n2 + n3 + n4 + n5 + n6)
```

So there are only **6 numbers with freedom**, and that is where the 6 comes from. In general, splitting N channels among S states gives

```
C(N + S - 1, S - 1)    ->    here S = 7, so C(N + 6, 6)
```

## Stars and bars

![Stars and bars for 3 channels and 7 states](stars_bars.png)

The rule has a picture. Write N stars for the N channels. Then write 6 bars to make 7 boxes between them. A string of stars and bars describes exactly one count vector.

With 3 channels and 7 states, the string is 3 stars plus 6 bars, so 9 symbols total. Choosing where the 6 bars go decides everything, and the rest are automatically stars:

```
number of strings = C(3 + 6, 6) = C(9, 6)
```

## Worked: 3 channels

```
C(9, 6) = C(9, 3)                 (picking 6 is the same as leaving 3)
        = 9! / (3! * 6!)
        = (9 * 8 * 7) / (3 * 2 * 1)
        = 504 / 6
        = 84
```

So with up to 3 channels, the hidden state space has **84** configurations.

## Worked: 5 channels

```
C(11, 6) = C(11, 5)
         = 11! / (5! * 6!)
         = (11 * 10 * 9 * 8 * 7) / (5 * 4 * 3 * 2 * 1)
         = 55440 / 120
         = 462
```

So 5 channels give **462** configurations.

## Sanity checks

| Channels N | Formula | Value | Does it make sense? |
|---|---|---|---|
| 1 | C(7, 6) | 7 | one channel sits in one of 7 states: 7 possibilities |
| 2 | C(8, 6) | 28 | both in the same state: 7 ways. In different states: C(7,2) = 21. Total 28 |
| 3 | C(9, 6) | 84 | as worked above |
| 4 | C(10, 6) | 210 | |
| 5 | C(11, 6) | 462 | as worked above |

![Hidden configurations per channel count](state_counts.png)

The thesis claimed factorial HMMs were too expensive for multi-channel data. The chart shows why that claim does not hold at the scale of real patches: even 5 channels gives only 462 configurations, which a laptop handles instantly.

## The exact decoder

# The exact decoder

![A detective with a magnifying glass and a rulebook](detective.png)

If you know the rulebook (the rates), the noise, and N, you can compute the best possible answer with plain probability. That is the exact decoder. No training, no guessing.

## Forward-backward in words

![Forward and backward passes](forward_backward.png)

The algorithm walks through the recording twice.

- The **forward pass** moves left to right. At each moment it asks: how likely is everything I have seen so far, and that the hidden state right now is this one?
- The **backward pass** moves right to left. At each moment it asks: how likely is everything that comes later, given that the hidden state right now is this one?

Multiply the two answers and normalize, and you get the probability of each hidden state at each moment **using the entire recording**. That is called smoothing. It is why the decoder does not panic at a single noisy sample: the past and future vote too.

![Each open count predicts a different current distribution](emissions.png)

The other ingredient is the emission. For each possible open count, the model predicts a distribution of currents: a bump around 1.74 when nothing is open, around 2.56 when one is open, and so on. The wider and more overlapping the bumps, the harder the problem. The decoder multiplies three things at each step: how well the reading matches each state, how likely each state is given the previous state, and the evidence from the future.

## A tiny worked example

Say the recording reads 2.56 at some moment.

- If 1 channel is open, the clean level is exactly 2.56. Very likely.
- If 2 are open, the clean level is 3.38. The reading is 0.82 away: possible but less likely.
- If 0 are open, the clean level is 1.74, also 0.82 away in the other direction.

So "1 open" wins this reading. The Markov part adds another voice: if the previous moment was "1 open" and openings cluster in time, staying at 1 is more plausible than jumping to 2 and back.

## Choosing N

The same forward pass gives a single number for a whole recording: the probability of the data under a given N. Run it for N = 1, 2, 3, ... and pick the N with the highest probability. That is model selection, and it needs no separate counting network.

## What it scores

The exact decoder was tested on 600 frozen recordings:

| Noise level | Moments correct | Channel count correct |
|---|---|---|
| normal | 96.3% | 100% |
| doubled | 83.4% | 100% |
| quadrupled | 67.2% | 100% |

It also scored 86.5% on patches with 4 or 5 channels, a case the neural models never saw. And it needed zero training. When the model is right, this is the ceiling: no learning algorithm can beat it on average.

## The catch

# The catch

![A locked chest with no key](lockbox.png)

The exact decoder has a catch, and it is a big one. It only works if you hand it the exact rules:

- the kinetic rates of the specific channel,
- the exact noise distribution,
- and N itself.

Real recordings arrive with none of these. The rates differ between channel types and even between experiments. The noise is never exactly the fitted shape. N is unknown; figuring it out is one of the two questions we started with.

There is a second, subtler issue. Simulators are simplified versions of cells. A method that is perfect for the simulator can still be wrong for the lab. So we want a method that learns the messy parts from data, while keeping the parts we are confident about.

That is the whole motivation for the next chapters: **keep the physics, learn the rest.**

## KI-HMM: the idea

# KI-HMM: the idea

![A bridge between gears and glowing nodes](bridge.png)

The plan in one sentence: take the exact decoder, keep the transition machinery as fixed physics, and replace the hand-fitted emission model with a small neural network. Add a second small network that learns to count the channels.

Why this order? Because each piece solves the previous piece's problem:

1. The exact decoder needs a rulebook. **A neural network can learn the rulebook** by watching examples.
2. A raw neural network, though, predicts each moment independently. It would produce flickering nonsense that violates how channels actually behave. **The fixed kinetic chain fixes that** by enforcing the transition rules.
3. The chain needs to know N. **A learned counter supplies it**, watching the whole recording.
4. A counter could cheat if it peeked at anything conditioned on N, so it is **deliberately placed before the N knob**. Details later.

We call the result **KI-HMM**: a kinetics-informed neural hidden Markov model.

![Architecture](architecture.png)

The diagram shows the flow. Blue blocks are learned by gradient descent. The green block is physics, with zero weights. The signal goes in on the left, the answer comes out on the right.

The whole model has about **136 thousand weights**. For comparison, a small language model has billions. This fits on a gaming laptop and trains in about half an hour.

## The eyes

# The eyes

![A magnifying glass over a line](magnifier.png)

The first learned block reads the raw signal. Its job: at every moment, say what the neighbourhood looks like. A channel opening has a signature: a step up, held for a while, then a step down. The eyes learn to spot these patterns.

## Step 0: normalization

Each recording is one vector of 1000 numbers. Before anything else, we subtract the average and divide by the standard deviation. Why? Because a real recording might be scaled differently by the amplifier. Shape matters; units should not. Two summary numbers, the average and the spread, are saved and passed along, because they carry useful information about N.

## Six zoom blocks

The eyes are six identical blocks in a row. Each block is:

```
convolution -> GELU -> 1x1 convolution -> add the block's input back
```

A **convolution** slides a small filter along the signal. The filter is 5 samples long, and it lights up when it sees the pattern it was tuned for. **GELU** is a smooth on-off function. The 1x1 convolution mixes information between filters. Adding the input back is called a **residual connection**: the block only learns a small correction rather than the whole transformation, which makes training easier.

![A learned robot reading a book](robot.png)

## Dilation: seeing far without paying for it

![Dilation](receptive.png)

Plain convolutions only see 5 samples at a time. Channel openings last much longer, so the network needs context. Instead of making the filter longer, we **space out** its taps. Block one reads neighbours 1 sample away, block two skips every other sample, then 4, 8, 16, 32. Same number of weights, much wider view.

By the last block, one output is influenced by 253 input samples. The arithmetic:

```
1 + 4*(1+2+4+8+16+32) = 1 + 4*63 = 253 samples = about 2.5 seconds
```

Why 4 per unit of dilation? A kernel of length 5 spans 4 gaps. Each block with dilation d adds 4d samples to the view. Same filter size, bigger context, no extra cost.

After the six blocks, every moment is described by **64 numbers**. Call it a fingerprint: a compact description of what the signal is doing around that moment.

## Why not a transformer or an LSTM?

You could use either. We chose dilated convolutions because they are small, fast, and easy to train on 1000-step sequences. The physics layer, coming later, supplies the long-range consistency. The eyes only need local context.

## The knob and the counter

# The knob and the counter

![A volume knob on a panel](knob.png)

Two questions need two answers: "what does the signal look like if N is 3?" and "what is N?". The architecture gives each question its own small block.

## The N knob (FiLM)

The same recording should be read differently depending on N. With one channel, a current of 2.56 means one open channel. With three channels, the same reading is ambiguous between different mixes.

So the network is told which N to assume, then adjusts:

```
e = learned embedding of N
c = join(e, average, spread)
scale, shift = small linear layer on c
fingerprint' = fingerprint * (1 + scale) + shift
```

This trick is called **FiLM** (feature-wise linear modulation). It is a volume knob: the same song, adjusted. Its value: one network serves all channel counts instead of five separate networks.

Why not simply concatenate N to the input? You could, and it works worse here, because the adjustment needs to be multiplicative: the interpretation of a current reading scales with N, it does not just add a bias.

## The counter

![An abacus](tally.png)

The counter is a small two-layer network. It looks at the average fingerprint, plus the average and spread of the raw signal, and outputs five scores: N = 1, 2, 3, 4 or 5. The winner is the answer.

The subtle part is **what it looks at**. The counter reads the fingerprint **before the N knob is applied**. Why? Because the knob receives N as input. If the counter read the adjusted fingerprint, it could copy N straight from the conditioning signal instead of learning to infer it from the data. We know this from experience: the first version of the model did exactly that. Its counting loss dropped to nearly zero while its accuracy on new data stayed at 35 percent. Moving the counter before the knob fixed it and took the count accuracy to 100 percent.

One lesson worth keeping: a network will find the easiest path to lower the loss, including paths you did not intend. Design and test for leaks.

## The rulebook layer

# The rulebook layer

![The rulebook flowchart](lumping.png)

The eyes produce a score per moment. Those scores are not the final answer, because they ignore time. A channel that opens for one sample and closes the next is possible in the scores, but nearly impossible under real kinetics: opening and closing takes time.

So the scores go through the **kinetic chain**: the same forward-backward machinery as the exact decoder, with two differences.

1. It uses the network's scores as the per-moment evidence instead of a hand-written noise formula.
2. To keep training fast, it works on a smaller chain: states are the open count (0 to N), not all 84 or 462 configurations.

## Why the smaller chain is an approximation

The full chain tracks all seven counts. The small chain tracks only how many channels are open. As we saw in the toy example, different hidden setups can share an open count but behave differently, so collapsing loses information. To minimize the damage, the transitions of the small chain are computed by **stationary-weighted averaging**:

```
for each pair (i, j) of open counts:
    P(i then j) =
        average over hidden states with open count i,
        weighted by how often each state occurs,
        of the probability of moving to any state with open count j
```

In words: group the 84 configurations into 4 buckets by their open count, then average the transition odds within each bucket. It is a controlled approximation, and the exact decoder remains available when the full chain is needed.

## Why this layer must be differentiable

The loss is computed after this layer, and gradients flow back through it. That is how the eyes learn what "useful" scores look like: not just accurate per-moment opinions, but opinions the physics can clean up into a correct sequence. The forward-backward recurrence is smooth, so backpropagation through 1000 steps just works. This is called backpropagation through time when it runs over sequences.

The layer has **zero weights**. It never learns. It only enforces rules. That is the difference between this model and a pure neural network: part of the computation is guaranteed correct by construction.

## Training: three report cards

# Training: three report cards

![Three blank report cards on a board](reportcards.png)

The model improves through three grades, combined into one loss.

![The three losses](losses.png)

**1. Smoothed answers against truth (weight 1.0).** The main grade. After the kinetic chain smooths the scores, compare the smoothed answer at every moment with the true open count, and penalize wrong probabilities. Because this grade is computed after the physics layer, it teaches the eyes to produce scores the rulebook can fix.

**2. Raw opinions against truth (weight 0.3).** A nudge. Even before smoothing, the per-moment scores should be sensible. This keeps the eyes useful on their own and speeds up learning.

**3. Channel count against truth (weight 0.5).** Trains the counter.

Why not just grade the final answer? Because the physics layer could do most of the work while the eyes stay weak, and then the whole model would be fragile. The extra grades keep every part honest.

## The data

3000 simulated recordings. Each has:

- N chosen uniformly between 1 and 5,
- one random noise level between 1 and 4 times the normal strength.

Random noise matters. If every training recording had the same noise, the network could memorize it. Randomizing forces it to read the noise level from the signal itself, which is exactly what is needed on real data.

## The settings

Adam optimizer, learning rate 0.001 with cosine decay (large steps early, fine steps late), gradients clipped at 1.0 so one bad batch cannot wreck the weights, batch size 32, 30 passes over the data. About 70 seconds per pass on a GTX 1660 Super.

## What could go wrong (and did)

- **Masking with minus infinity.** Impossible open counts are masked out of the scores. Using negative infinity makes PyTorch's logsumexp backward produce NaN, which poisons all weights. We use a large finite negative number instead, minus one million.
- **The counter leak**, described earlier: 35 percent accuracy until the counter was moved before the N knob.
- **The open noise is not negligible.** Early on, we treated the open level as a clean number because its scale parameter is tiny. But the distribution has heavy tails: its actual spread is about 0.23 per channel, comparable to the closed state. Likelihood-based decoders must convolve both distributions exactly.

## Inference: one pass

# Inference: one pass

![Inference pipeline](inference.png)

Once trained, answering a new recording takes one pass.

1. The counter reads the recording and outputs N. On the frozen test set, this is correct 100 percent of the time.
2. The encoder and the N knob produce scores for open counts 0 to N.
3. The kinetic chain smooths the scores over the whole recording, using past and future.
4. The final answer is the most likely open count at each moment.

## The N shortcut we tried, and why it failed

There is a principled alternative for step 1: run the model at every candidate N and pick the one with the highest marginal likelihood, exactly like the exact decoder does. We tried it. It scored about 30 percent.

The reason is subtle. The scores are normalized per moment across the open counts. Different N gives a different set of possible counts, so the normalization constants differ between candidates. Comparing their likelihoods is then apples to oranges. The learned counter sidesteps the problem by making N prediction its own supervised task. This is a good example of a mathematically elegant idea losing to a practical one; the honest write-up keeps both numbers.

## Results

# Results

![Leaderboard](leaderboard.png)

Every model was retrained and tested on the same frozen set of recordings, at three noise levels. That matters: the thesis's original numbers let the test set double as the validation set, which flatters every model.

| Model | normal | 2x noise | 4x noise | Channel count |
|---|---|---|---|---|
| Thesis LSTM | 81.2% | 57.1% | 32.4% | not measured |
| Thesis CCNN + LSTM | 89.2% | 70.8% | 38.3% | 96.8% |
| KI-HMM (new) | 93.3% | 81.6% | 63.9% | **100%** |
| Exact decoder (new) | **96.3%** | **83.4%** | **67.2%** | **100%** |

KI-HMM beats the thesis's best model by 4 points at normal noise, 11 points at double noise, and 25 points at quadruple noise. It lands within about 3 points of the exact decoder, which is handed the true N, the true rates, and the true noise shape. KI-HMM is handed none of those.

Transitions, the moments right around an opening or closing, are where smoothing pays off:

| | normal | 2x | 4x |
|---|---|---|---|
| Thesis CCNN + LSTM | 82.9% | 64.3% | 36.3% |
| KI-HMM | 87.1% | 69.6% | 51.9% |

## Watching it work

![Predictions against the raw signal](kihmm_predictions.png)

Light blue is the raw recording, green dashed is the truth, red is KI-HMM, and dotted blue is the exact decoder given the true N. On these four recordings, KI-HMM counted the channels correctly by itself and scored 95.4, 90.6, 99.4 and 91.5 percent of moments correct.

![A transition close-up](kihmm_prediction_zoom.png)

The close-up shows a single opening. The red line lags the green step by a sample or two, then locks on. That delay is the price of honesty: the model uses only the signal, never the answer.

## Where it does not win

![Honesty on a scale](scale.png)

We also built a sabotage test: correlated noise, filtered signals, drifting rates. The exact decoder stays the most robust, and KI-HMM follows a few points behind. The lesson: when the model matches reality, math beats learning. Learning wins when reality does not hand you the model.

## What is new, and what comes next

# What is new, and what comes next

Three ingredients had not been combined before for multi-channel patch-clamp data:

1. **Learning the emission model** with a neural network, instead of fitting noise by hand.
2. **Keeping the exact kinetic chain** for several channels. The thesis dismissed this as too expensive; for 1 to 5 channels it is tiny.
3. **Counting the channels with a learned head**, inside the same model, instead of a separate network and a pile of heuristics.

The thesis listed exactly this direction as future work: put biophysics into the deep learning. This is that, with measurements.

What comes next:

- Run both methods on the real recordings, including the glibenclamide data, where kinetics change and noise is far from the simulator's.
- Let the model adapt the kinetic rates, not only the emissions, when real data disagrees with the simulator. The exact decoder can do this with expectation-maximization, which finds parameters without labels.
- Use the exact decoder as a referee on lab data: fit it to the recording, then compare its answer with the neural model's. When both agree, confidence is high; when they disagree, the disagreement is informative.

The one-line summary: a hand-built physics decoder is the best method on simulated data, a learned model with physics inside gets within a few points while knowing almost nothing in advance, and on real recordings, where the rulebook is missing, that trade looks like the right one.

## Math appendix

# Math appendix

The worked details behind the shortcuts used in the book.

## Factorials and choose

```
4! = 4 * 3 * 2 * 1 = 24
0! = 1
C(a, b) = a! / (b! * (a - b)!)
C(5, 2) = 120 / (2 * 6) = 10
```

## Stars and bars, step by step

To split N channels among 7 states, write N stars and 6 bars. Example with N = 3:

```
* * * | | | | | |          -> (3,0,0,0,0,0,0)
* | * | * | | | |          -> (1,1,1,0,0,0,0)
* | | | | | | *            -> (1,0,0,0,0,0,2)
```

Every arrangement is exactly `N + 6` symbols long, and choosing the positions of the 6 bars fixes the split:

```
possibilities = C(N + 6, 6)
N = 3:  C(9, 6)  = 9!/(6! 3!)  = 504/6     = 84
N = 5:  C(11, 6) = 11!/(6! 5!) = 55440/120 = 462
```

## Convolution, with dice

![Dice distributions](convolution.png)

The observation is a sum of independent noises, and the distribution of a sum is the convolution of the distributions. The dice picture: one die is flat; the sum of two dice is triangular. Each total can be made in several ways, and convolution counts those ways. For the channels, the same idea combines k open-channel noises and N minus k closed-channel noises into one predicted distribution. We compute it with FFTs, which are just a fast way to convolve.

## Matrix exponential, worked on two states

Take a channel with just open and closed. Opening rate 5 per second, closing rate 10 per second, step 0.01 seconds. The linear guess:

```
open -> closed: 10 * 0.01 = 0.10
closed -> open:  5 * 0.01 = 0.05
```

The exact one-tick probabilities from `expm(dt R)` are 0.093 and 0.046. They are slightly smaller because some channels would have hopped twice within the step, and those double hops bring them back. `expm` accounts for all of it. With these numbers the difference is small, but across 1000 steps the errors would pile up.

## logsumexp

Probabilities become tiny after multiplying many of them, so we work with their logarithms and add instead. Adding logs is not the same as the log of the sum, so we use:

```
logsumexp(a, b) = max(a, b) + log(1 + exp(-|a - b|))
```

Example: `logsumexp(-1000, -1001) = -1000 + log(1 + e^-1) = -999.687`. The max trick keeps the exponential from underflowing. Every `logsumexp` in the forward-backward code is this identity applied over many terms.

## Bayes in one line

```
posterior(state)  proportional to  prior(state) * likelihood(data | state)
```

The prior comes from the transition matrix; the likelihood comes from the emission model; the normalizer makes the probabilities sum to one. Forward-backward is this rule applied at every moment efficiently, without redoing work.

## What the lumping formula does

For the small chain used inside KI-HMM, transitions are averaged within buckets of equal open count, weighted by how often each hidden configuration occurs at stationarity:

```
P_small(i -> j) =
   sum over hidden states s with open count i of
        pi(s) * P(s moves to a state with open count j)
   divided by
   sum over hidden states s with open count i of pi(s)
```

It is a projection: correct on average, not exact. That is why the exact decoder keeps the full chain.

## Glossary

# Glossary

**HMM**: a model with hidden states, rules for switching between them, and rules for what you observe.

**Emission**: what you expect to measure while in a given hidden state.

**Forward-backward**: the algorithm that turns a sequence of measurements into probabilities for every hidden state at every moment.

**Smoothing**: using past and future data together to estimate the present. Reduces flickering.

**Bayes-optimal**: the best possible accuracy if your model of the world is exactly right.

**Convolution**: the math for adding independent random quantities; spreads one distribution by another.

**Dilation**: spacing out a filter's taps so it covers more input without more weights.

**Residual connection**: adding a layer's input back to its output, so the layer learns a correction.

**FiLM**: using one input to compute a scale and shift for another layer's features.

**Amortized**: paying a cost once during training so new data can be processed in one pass.

**Trans-dimensional**: inference that decides how many components exist, not just their values.

**Stationary distribution**: the long-run fraction of time a chain spends in each state.

**Marginal likelihood**: the probability of the whole recording under a candidate model, with hidden states summed out.
