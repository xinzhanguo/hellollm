from transformers import pipeline, set_seed
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import GPT2TokenizerFast

# save model dir
save_path = "./shakespeare"

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.normalizer = Sequence([NFKC()])
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()
special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
trainer = BpeTrainer(vocab_size=50000, show_progress=True,
                     inital_alphabet=ByteLevel.alphabet(), special_tokens=special_tokens)
files = ["text/romeo_and_juliet.txt"]
tokenizer.train(files, trainer)
newtokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
newtokenizer.save_pretrained(save_path)
# load tokenizer from pretrained
tokenizer = GPT2Tokenizer.from_pretrained(save_path)
tokenizer.add_special_tokens({"eos_token": "</s>", "bos_token": "<s>",
                             "unk_token": "<unk>", "pad_token": "<pad>", "mask_token": "<mask>"})
# creating the configurations from which the model can be made
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id
)
# creating the model
model = GPT2LMHeadModel(config)
# setting train data
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./text/romeo_and_juliet.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, mlm_probability=0.15
)
# setting train args
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_gpu_train_batch_size=16,
    save_steps=2000,
    save_total_limit=2,
)
# start train
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
trainer.train()
model.save_pretrained(save_path)
# test model
generator = pipeline('text-generation', model=save_path)
set_seed(13)
txt = generator("Hello", max_length=10)
print(txt)
