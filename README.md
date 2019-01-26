# AbsMeetingSummarization
Abstractive Meeting Summarization

## 1 Dataset
This dataset is extracted from [AMI meeting](http://groups.inf.ed.ac.uk/ami/download/), the dataset folder is organized as follows:
```
data/
    ami-dataset.pkl  # AMIDataset instance
    ami-vocab.pkl    # Vocabulary
    train.json       # training data
    valid.json       # validation data
    test.json        # testing data
```
The description of records in ```*.json``` is as follows:
```json
{
    "agents": "Who is the speaker in one sentence", // list of char
    "st_times": "Start time of one sentence", // list of float
    "end_times": "End time of one sentence",  // list of float
    "article_sents": "Sentences of this meeting", 
    "article_pos_tags": "Pos tags for each token",
    "article_ner_tags": "Ner tags for each token", 
    "article_tf_tags": "Tf tags for each token", 
    "article_idf_tags": "IDF tags for each token", 
    "summary_sents": "Sentences of summary of this meeting", 
    "summary_pos_tags": "Pos tags for each token for each summary",
    "summary_ner_tags": "Ner tags for each token for each summary"
}
```

## 2. Requirements
Need install rouge in your linux environment[link](https://blog.csdn.net/wr339988/article/details/70165090).  And then 
```
$ pip install pyrouge
```


## 2 Training
```bash
cd AbsMeetingSummarization
bash scripts/run_HAS.sh
```

## 3 Testing
Build a directory named ```ckp`` and move your saved model into it. The directory may look like:
```
ckp/
    models/
        your_model.pt
```

```bash
python test.py --ckp-path checkpoints/param_tuning/ --report_score
```