# BRENT: Bidirectional Retrieval Enhanced Norwegian Transformer

This is the repository for the NoDaLiDa 2023 paper on developing a Norwegian retrieval-augmented langauge model. The pre-print can be found at [this url](https://arxiv.org/abs/2304.09649).

## Abstract
Retrieval-based language models are increasingly employed in question-answering tasks. These models search in a corpus of documents for relevant information instead of having all factual knowledge stored in its parameters, thereby enhancing efficiency, transparency, and adaptability. We develop the first Norwegian retrieval-based model by adapting the REALM framework and evaluate it on various tasks. After training, we also separate the language model, which we call the _reader_, from the retriever components, and show that this can be fine-tuned on a range of downstream tasks. Results show that retrieval augmented language modeling improves the reader's performance on extractive question-answering, suggesting that this type of training improves language models' general ability to use context and that this does not happen at the expense of other abilities such as part-of-speech tagging, dependency parsing, named entity recognition, and lemmatization.

## People

BRENT was a joint work between Lucas Georges Gabriel Charpentier, Sondre Wold, David Samuel and Egil Rønningstad.

## Citing

If you publish work that uses or references the data, please cite our [NODALIDA article](). BibEntry:

```
@InProceedings{charpentier-etal-2023brent,
  {Charpentier, Lucas Georges Gabriel and Wold, Sondre and Samuel, David and R{\o}nningstad, Egil},
  title = {BRENT: Bidirectional Retrieval Enhanced Norwegian Transformer},
  booktitle = {Proceedings of the 24th Nordic Conference on Computational Linguistics},
  year = {2023},
  address = {Torshavn, Pharoe Islands},
}
```
