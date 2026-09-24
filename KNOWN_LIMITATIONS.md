# Known limitations

1. **The exact controlled-experiment orchestration code is not included.** The repository includes the final machine-readable controlled results and a result-regeneration utility.
2. **The exact theorem-grid point coordinates used for the reported 3,888-point check are unavailable.** The reported JSON records the sample count and worst checks; a separate independent 3,888-point executable check is included.
3. **The frozen ModernBERT embedding cache is not committed.** The repository records the pinned model revision and embedding-extraction configuration so embeddings can be regenerated, subject to software/hardware reproducibility limits.
4. **Source/raw outputs for the supplementary wrong-vote-location intervention and secondary pretrained-synthetic compatibility control are unavailable.** Their published descriptions/figures remain in the authoritative supplement.
5. **Raw WOS-46985 data are not redistributed.** The experiment accepts a local copy and contains public-source acquisition fallbacks.
6. **No exact dependency lock or author-selected software license is included.** `requirements.txt` records the runtime libraries used by the included scripts.
