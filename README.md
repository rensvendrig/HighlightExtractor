# pysumm
A query-focused multi-document summarization pipeline

We use the common extractive summarization framework, i.e., sentence
ranking followed by sentence selection.
In sentence ranking, we implement query-sensitive LexRank and Manifold.
In sentence selection, we implement the n-gram redudancy measurement.

# usage
See example.py

# To do
Sentence ranking: Support Vector Regression
Sentence selection: Submodular and Integer Linear Programming
