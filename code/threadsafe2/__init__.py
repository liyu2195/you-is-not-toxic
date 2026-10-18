"""THREADSAFE: valid conversation-level metrics from LLM labels and a few hundred gold labels.

  units     Algorithm 1, "at least one" shares over conversation units (threads, articles, prefixes, users)
  contrast  Algorithm 2, structural contrasts (depth, size)
  pairs     Algorithm 3, the reply association
"""
from . import units, contrast, pairs, intervals  # noqa: F401
