
# Vietnamese tone restoration

![logo](./assets/icon.png)

A Natural Language Processing project for restoring Vietnamese sentence tone.


## Requirements 

```bash 
torch, torchtext, pandas, numpy, tqdm, matplotlib (newest version)
```
    
## Documentation

  Current solution:
  
  - GRU encoder decoder with beam search
  - Transformers with beam search

Result training on 100K sentences (evaluate on 1K sentences)
  
  - GRU encoder decoder: 0.712 (~40 minutes training)
  - Transfomers: **0.742** (~11 minutes training)
  - N-Gram: 0.722 (~2h 16min inference)
  - 
Result training on 200K sentences (evaluate on 500 sentences)
  
  - GRU encoder decoder: 0.66 (~140 minutes training)
  - Transfomers: *updating* 
  - N-Gram: 0.813 (~2h 3min inference)
  - One shot BASE Transformers: 0.818

Result training on 2M senteces (evaluate on 1K sentences)
  - Transfomers (BASE like in the paper): **0.937** (~20 hours training on GTX 2080)

*updating...*

## Authors

- [@HKAB](https://www.github.com/HKAB)
- [@tacbliw](https://github.com/tacbliw)
- [@xuantruong2000](https://github.com/xuantruong2000)
- [@ultoxtung](https://github.com/ultoxtung)

## References

- [d2l.ai](https://d2l.ai/) (most codes are borrowed from here)
- [Veritable Tech Blog](https://blog.ceshine.net/post/implementing-beam-search-part-1/) (beam search idea)
- [Language Translation with TorchText](https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html)
