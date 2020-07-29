Scope of Negation 
===================================================================

## Requirements
Python 3.5+ (recommended: Python 3.6) \
Python packages: list of packages are provided in ./env-setup/requirements.txt file. \
Embedding: Download the file "glove.6B.300d.txt" from https://nlp.stanford.edu/projects/glove/ and put it in ./embeddings/globe directory

```bash
# Create virtual env (Assuming you have Python 3.5 or 3.6 installed in your machine) -> optional step
python3 -m venv your_location/scope-of-negation
source your_location/scope-of-negation/bin/activate

# Install required packages -> required step
pip install -r ./env-setup/requirements.txt
python -m spacy download en_core_web_sm
export HDF5_USE_FILE_LOCKING='FALSE'
```


## How to Run

- Example command line to train the cue-detector: 
```bash
  python train.py -c ./config/config.json 
```
  + Arguments:
	  - -c, --config_path: path to the configuration file, (required)
  
  *Note that: Training the scope detector is optional, a trained model is already provided in "./model" directory. Below step shows how to use the pre-trained model to predict scope of negation of the sentences in given text file. The file should have one sentence per line.
	
- Example command line to apply prediction on a given text file. 
```bash
  python predict.py -c ./config/config.json -i ./data/sample-io/input_file.txt -o ./data/sample-io/ 
```
  + Arguments:
	  - -c, --config-path: path to the configuration file; (required). Contains details parameter settings.
	  - -i, --input-file-path: path to the sample input file (text file, one sentence per line); (required)
	  - -o, --output-dir: path to the output directory (output file is created in this directory); (required)
	  - --cd_sco_eval: if true, then creates a prediction file (in "./data/cd-sco-prediction/" by default) to evaluate cd-sco test corpus, (optional)
  
## Input and Output Files
- Sample input file:   \
./data/sample-io/input_file.txt (input file must contain one sentence per line) \
Sample sentence: \
"I don't mean to be glib about your concerns, but if I were you, I might be more concerned about the near-term rate implications of this $1.	"

- Sample output files: \
./data/sample-io/output_file.txt (contains sentences with negation, one sentence per line) \
Sample output: \
"I/I_S do/I_S n't/I_C mean/I_S to/I_S be/I_S glib/I_S about/I_S your/I_S concerns/I_S ,/O_S but/O_S if/O_S I/O_S were/O_S you/O_S ,/O_S I/O_S might/O_S be/O_S more/O_S concerned/O_S about/O_S the/O_S near/O_S -/O_S term/O_S rate/O_S implications/O_S of/O_S this/O_S $/O_S 1/O_S ./O_S" \
(I_S: inside scope of negation, O_S: outside scope, I_C: negation cue)

## Citation

Predicting scope of negation is part of the paper "Predicting the Focus of Negation: Model and Error Analysis". 
```bibtex
@inproceedings{hossain-etal-2020-predicting,
    title = "Predicting the Focus of Negation: Model and Error Analysis",
    author = "Hossain, Md Mosharaf  and
      Hamilton, Kathleen  and
      Palmer, Alexis  and
      Blanco, Eduardo",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.743",
    pages = "8389--8401",
    abstract = "The focus of a negation is the set of tokens intended to be negated, and a key component for revealing affirmative alternatives to negated utterances. In this paper, we experiment with neural networks to predict the focus of negation. Our main novelty is leveraging a scope detector to introduce the scope of negation as an additional input to the network. Experimental results show that doing so obtains the best results to date. Additionally, we perform a detailed error analysis providing insights into the main error categories, and analyze errors depending on whether the model takes into account scope and context information.",
}
```
