
# Vietnamese tone restoration :newspaper:

A Natural Language Processing project for restoring Vietnamese sentence tone.


## Requirements 

```bash 
torch, torchtext, pandas, numpy, tqdm, matplotlib (newest version)
```
    
## Documentation

  Current solution:
  
  - GRU encoder decoder with beam search
  - Transformers with beam search

Temperary result training on 100K sentences (evaluate on 1K sentences)
  
  - GRU encoder decoder: 0.712 (~40 minutes training)
  - Transfomers: **0.742** (~11 minutes training)

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
