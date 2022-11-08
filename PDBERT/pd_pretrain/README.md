### Installation
First clone repository, open a terminal and cd to the repository
    
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    mkdir -p data/raw/semeval2014  # creates directories for data
    mkdir -p data/transformed
    mkdir -p data/models
    

For downstream finetuning, you also need to install torch, pytorch-transformers package and APEX (here for CUDA 10.0, which
is compatible with torch 1.1.0 ). You can also perform downstream finetuning without APEX, but it has been used for the paper.

    pip install scipy sckit-learn  # pip install --default-timeout=100 scipy; if you get a timeout
    pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
    pip install pytorch-transformers tensorboardX

    cd ..
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    
### Preparing data for BERT Language Model Finetuning

We make use of two publicly available research datasets
for the domains laptops and restaurants:

* Amazon electronics reviews and metadata for filtering laptop reviews only:
    * Per-category files, both reviews (1.8 GB) and metadata (187 MB) - ask jmcauley to get the files, 
    check http://jmcauley.ucsd.edu/data/amazon/amazon_readme.txt
* Yelp restaurants dataset:
    * https://www.yelp.com/dataset/download (3.9 GB)
    * Extract review.json

Download these datasets and put them into the data/raw folder.

To prepare the data for language model finetuning run the following python scripts:

    python prepare_laptop_reviews.py
    python prepare_restaurant_reviews.py
    python prepare_restaurant_reviews.py --large  # takes some time to finish

Measure the number of non-zero lines to get the exact amount of sentences
    
    cat data/transformed/restaurant_corpus_1000000.txt | sed '/^\s*$/d' | wc -l
    # Rename the corpora files postfix to the actual number of sentences
    # e.g  restaurant_corpus_1000000.txt -> restaurant_corpus_1000004.txt

Concatenate laptop corpus and the small restaurant corpus to create the mixed corpus (restaurants + laptops)

    cd data/transformed
    cat laptop_corpus_1011255.txt restaurant_corpus_1000004.txt > mixed_corpus.txt



## LM Finetuning

The LM finetuning code is an adaption to a script from the huggingface/pytorch-transformers repository:
* https://github.com/huggingface/pytorch-transformers/blob/v1.0.0/examples/lm_finetuning/finetune_on_pregenerated.py

Prepare the finetuning corpus, here shown for a test corpus "dev_corpus.txt":

    python pregenerate_training_data.py \
    --train_corpus lap_rest_mixed_corups.txt \
    --bert_model bert-base-uncased --do_lower_case \
    --output_dir dev_corpus_prepared_dp_mams/ \
    --epochs_to_generate 2 --max_seq_len 256


Run actual finetuning with:

    python finetune_on_pregenerated.py \
    --pregenerated_data dev_corpus_prepared_dp_lap/ \
    --bert_model bert-base --do_lower_case \
    --output_dir dev_corpus_finetuned_dp_lap/ \
    --epochs 4 --train_batch_size 16