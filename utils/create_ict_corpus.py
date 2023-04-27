import argparse
import stanza
import json
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess raw wiki dump into doc sections for ict")
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/nno/nnwiki-20201201.txt",
        help="Path to the raw wiki dump"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/ict_warmup.json",
        help="Path to the raw wiki dump"
    )

    parser.add_argument(
        "--size",
        type=int,
        default=10000,
        help="Number of documents in the produced corpus"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    nlp = stanza.Pipeline(lang='no', processors='tokenize')
    processed_documents = []

    with open(args.data_path, encoding='utf8') as f:
        document = f.read()
        docs = document.split("Introduction") #Raw data file does not contain any metadata to split on... The Introduction queue seems to be the easiest way to seperate article from article
        print(len(docs))
        processed_docs = []
        for doc in tqdm(docs[2:args.size + 2]):
            d = doc.split("\n\n") #All header texts are ended by the first double line break of the article.
            introduction_text = d[0]
            sents = nlp(introduction_text)
            if len(sents.sentences) > 4: #Some articles have very short introductions... skip those. 
                processed_docs.append(introduction_text.strip().replace("\n", ""))
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_docs, f, ensure_ascii=False, indent=4)
