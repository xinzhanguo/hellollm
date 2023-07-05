# Hello LLM

[Chinese/中文](https://github.com/xinzhanguo/hellollm/blob/main/README_ZH.md)

new train a llm model.
not fine tuned model.

## how to train
docker
```
docker build -t hellollm:beta .
# use gpu
# docker run -it --gpus all hellollm:beta sh
docker run -it hellollm:beta sh
python hellollm.py
```
linux 
```
# create env
python3 -m venv ~/.env
# active env
source ~/.env/bin/activate
# download code
git clone git@github.com:xinzhanguo/hellollm.git
cd hellollm
# install requirements
pip install -r requirements.txt
# run train
python hellollm.py
```

## Detail

#### prepare data

First we need to prepare the data for training, we are training based on <Romeo and Juliet>.

#### train tokenizer

Tokenization is to divide the input text into meaningful subunits (tokens).
Through the following code, a new tokenizer based on our data

```
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import GPT2TokenizerFast

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.normalizer = Sequence([NFKC()])
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

special_tokens = ["<s>","<pad>","</s>","<unk>","<mask>"]
trainer = BpeTrainer(vocab_size=50000, show_progress=True, inital_alphabet=ByteLevel.alphabet(), special_tokens=special_tokens)
files = ["text/remeo_and_juliet.txt"]

tokenizer.train(files, trainer)

newtokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
newtokenizer.save_pretrained("./shakespeare")
```

ls shakespeare:
```
merges.txt
special_tokens_map.json
tokenizer.json
tokenizer_config.json
vocab.json
```

#### Training model

```
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("./shakespeare")
tokenizer.add_special_tokens({
  "eos_token": "</s>",
  "bos_token": "<s>",
  "unk_token": "<unk>",
  "pad_token": "<pad>",
  "mask_token": "<mask>"
})
config = GPT2Config(
  vocab_size=tokenizer.vocab_size,
  bos_token_id=tokenizer.bos_token_id,
  eos_token_id=tokenizer.eos_token_id
)
model = GPT2LMHeadModel(config)
from transformers import LineByLineTextDataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./text/remeo_and_juliet.txt",
    block_size=128,
)
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, mlm_probability=0.15
)
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_gpu_train_batch_size=16,
    save_steps=2000,
    save_total_limit=2,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
trainer.train()
model.save_pretrained('./shakespeare')

```

ls ./shakespeare and find added three files.
```
config.json
generation_config.json
pytorch_model.bin
```

#### test model

```
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='./shakespeare')
set_seed(42)
txt = generator("Hello", max_length=30)
print(txt)
```
