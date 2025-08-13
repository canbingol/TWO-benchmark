import sentencepiece as snp
from model import ModelArgs, Transformer
import torch
import torch.nn.functional as F
from datasets import load_dataset
import statistics
from time import perf_counter

dataset = load_dataset("canbingol/TWO-Bench", split="train")
tokenizer = snp.SentencePieceProcessor()
tokenizer.load('tokenizer.model')
device = "cuda" if torch.cuda.is_available() else  "cpu"

cfg = ModelArgs
model = Transformer(cfg)
checkpoint = torch.load('checkpoint_epoch_1_step_20000.pt')
model.load_state_dict(checkpoint['model_state_dict'])

dataset = load_dataset("canbingol/TWO-Bench", split="train")
model.eval()
def benchmarking(prefix, sample):
    prefix_ids = torch.tensor(tokenizer.encode(prefix))
    sample_ids = torch.tensor(tokenizer.encode(sample))
    len_prefix = len(prefix_ids)
    prefix_ids = prefix_ids.unsqueeze(0)
    sample_ids = sample_ids.unsqueeze(0)

    input_ids = torch.cat((prefix_ids,sample_ids), dim=-1)
    seq_len = input_ids.size(1)
    with torch.no_grad():
        logits = model(input_ids)

    logprobs = torch.log_softmax(logits[:,len_prefix-1:seq_len-1], dim=-1)
    cont_targets = input_ids[:,len_prefix:seq_len]


    llh_tokens = logprobs.gather(2,cont_targets.unsqueeze(-1)).squeeze(-1)
    total_llh = llh_tokens.sum(dim=1)
    mean_llh = -llh_tokens.mean(dim=1)
    return mean_llh.item()

total_acc = 0
acc_list = []
start = perf_counter()
for i,data in enumerate(dataset):
    prefix = data["prefix"]
    sample1 = data["sample1"]
    sample2 = data["sample2"]

    score1 = benchmarking(prefix, sample1)
    score2 = benchmarking(prefix, sample2)
    print(f"{i}th sample scores:\n")
    print(f"\tscore1: {score1}\n")
    print(f"\tscore2: {score2}\n\n\n")


    if score1 < score2: 
        acc_list.append(1) 
    else:
        acc_list.append(0)


print(f"model TWO score : {statistics.mean(acc_list)}")
end = perf_counter()
print(f"time: {end - start:.4f} second")
