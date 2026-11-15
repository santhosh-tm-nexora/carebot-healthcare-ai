# Week 6 Evidence Text

All three NER configurations tied at 1.0000 macro F1 and 1.0000 JSON
validity on the 30 synthetic test notes. Config 1 is selected as the best
configuration because it achieves the tied best score with the fewest epochs
(2) and lowest dropout (0.1), reducing training cost without a measured
accuracy trade-off.

The plain and entity-constrained summarization variants both used the local
template fallback and measured 1.0000 faithfulness. No Anthropic or Gemini API
key was configured, so no external LLM provider was evaluated.
