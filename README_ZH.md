# Hello LLM

[English](https://github.com/xinzhanguo/hellollm/blob/main/README.md)

从新训练一个大语言模型。
**注意是从新训练一个大语言模型，不是微调。**


## 运行代码

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

## 训练说明

#### 一、准备数据
首先我们要为训练准备数据，我们基于<罗密欧与朱丽叶>进行训练。

#### 二、训练分词器
分词(tokenization) 是把输入文本切分成有意义的子单元（tokens）。
通过以下代码，根据我们的数据一个新的分词器：
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

#### 三、训练模型
利用下面代码进行模型训练：
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
# 配置GPT2模型参数
config = GPT2Config(
  vocab_size=tokenizer.vocab_size,
  bos_token_id=tokenizer.bos_token_id,
  eos_token_id=tokenizer.eos_token_id
)
# 创建模型
model = GPT2LMHeadModel(config)
# 训练数据我们用按行分割
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
# 配置训练参数
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
# 保存模型
model.save_pretrained('./shakespeare')
```

成功运行代码，我们发现shakespeare目录下面多了三个文件:
```
config.json
generation_config.json
pytorch_model.bin
```

现在我们就成功生成训练了一个大语言模型。

#### 四、测试模型

我们用文本生成，对模型进行测试代码如下:
```
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='./shakespeare')
set_seed(42)
txt = generator("Hello", max_length=30)
print(txt)
```
