import json
import torch
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu,SmoothingFunction
from im2latex_dataset import Im2LatexDataset
from collate_trans import collate_transformer
from resnet_encoder import ResNetEncoder
from transformer_decoder import TransformerDecoder
from torch.utils.data import DataLoader
from hme100k_dataset import HME100kDataset
from self_collected_dataset import SelfCollectedDataset

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER=r"D:\APS360_proj\data\processed\tokenizer.json"
MAX_LEN=100

# #imlatex sets
# CSV=r"D:\APS360_proj\data\processed\im2Latex\test_clean_fixed.csv"
# IMAGE_DIR=r"D:\APS360_proj\data\raw\im2Latex\formula_images_processed"
CHECKPOINT = r"D:\APS360_proj\saved_paths\im2latex_transformer_best_after140epoch.pth"

# hme100 sets
# TXT = r"D:\APS360_proj\data\raw\HME100k\splits\test.txt"
# IMAGE_DIR = r"D:\APS360_proj\data\processed\hme100k_processed"
# CHECKPOINT = r"D:\APS360_proj\saved_paths\hme100k_finetuned_best_170epoch_final.pth"

#self collected
TXT = r"D:\APS360_proj\data\self_collected\self_labels.txt"
IMAGE_DIR = r"D:\APS360_proj\data\self_collected\self_images_processed"


with open(TOKENIZER,"r",encoding="utf-8") as f:
    vocab=json.load(f)["vocab"]

# check tokens
# with open(TXT,"r",encoding="utf-8") as f:
#     for line in f:
#         img,formula=line.strip().split("\t")

#         print("FORMULA:")
#         print(formula)

#         for t in formula.split():
#             if t not in vocab:
#                 print("UNKNOWN TOKEN:",t)

#         break


itos={v:k for k,v in vocab.items()}
PAD=vocab["<PAD>"]
BOS=vocab["<BOS>"]
EOS=vocab["<EOS>"]

#remove whitespace helper
def normalize_formula(tokens):
    return "".join(tokens).replace(" ","")

# dataset=Im2LatexDataset(CSV,IMAGE_DIR,TOKENIZER)
# dataset=HME100kDataset(
#     txt_path=TXT,
#     image_folder=IMAGE_DIR,
#     tokenizer_path=TOKENIZER
# )
dataset=SelfCollectedDataset(
    txt_path=TXT,
    image_folder=IMAGE_DIR,
    tokenizer_path=TOKENIZER
)

loader=DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_transformer
)

# for images,captions,padding_mask in loader:
#     print(images.shape)
#     break

encoder=ResNetEncoder(embed_dim=256).to(DEVICE)
decoder=TransformerDecoder(vocab_size=len(vocab),embed_dim=256,heads=8,layers=3).to(DEVICE)

checkpoint=torch.load(CHECKPOINT,map_location=DEVICE)
encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])

encoder.eval()
decoder.eval()
torch.backends.cudnn.benchmark=True

print("Loaded checkpoint:", CHECKPOINT)

@torch.no_grad()

def generate_greedy_batch(images):
    memory=encoder(images)
    batch_size=images.size(0)

    tokens=torch.full(
        (batch_size,1),
        BOS,
        device=DEVICE
    )

    finished=torch.zeros(batch_size,dtype=torch.bool,device=DEVICE)

    for _ in range(MAX_LEN):
        mask=tokens==PAD
        output=decoder(memory,tokens,mask)

        next_token=output[:,-1].argmax(-1,keepdim=True)
        tokens=torch.cat([tokens,next_token],dim=1)
        finished |= next_token.squeeze(1)==EOS

        if finished.all():
            break

    results=[]

    for seq in tokens:
        results.append(
            [itos[t.item()] for t in seq[1:] if t.item() not in [PAD,EOS]]
        )

    return results

@torch.no_grad()
def generate_beam_search(image,beam_size=3,alpha=0.1):
    memory=encoder(image)
    beams=[([BOS],0.0)]
    finished=[]

    for _ in range(MAX_LEN):
        candidates=[]

        for seq,score in beams:
            if seq[-1]==EOS:
                finished.append((seq,score))
                continue

            tokens=torch.tensor([seq],device=DEVICE)
            mask=tokens==PAD

            output=decoder(memory,tokens,mask)[:,-1]
            log_probs=torch.log_softmax(output,dim=-1)

            values,indices=torch.topk(log_probs,beam_size)

            for v,i in zip(values[0],indices[0]):
                candidates.append(
                    (seq+[i.item()],score+v.item())
                )

        if not candidates:
            break

        beams=sorted(
            candidates,
            key=lambda x:x[1]/len(x[0])**alpha,
            reverse=True
        )[:beam_size]

    finished+=beams

    best=max(
        finished,
        key=lambda x:x[1]/len(x[0])**alpha
    )[0]

    return [
        itos[t]
        for t in best[1:]
        if t not in [PAD,EOS]
    ]


# prediction cycle
predictions=[]
references=[]
norm_predictions=[]
norm_references=[]

correct=0
correct_norm=0


for images,captions,padding_mask in tqdm(loader):
    images=images.to(DEVICE)
    pred=generate_beam_search(images,beam_size=3,alpha=0.1)
    caption=captions[0]

    # print(images.shape) # testing
    # break

    ref=[
        itos[t.item()]
        for t in caption
        if t.item() not in [PAD,BOS,EOS]
    ]
    predictions.append(pred)
    references.append([ref])
    pred_norm=normalize_formula(pred)
    ref_norm=normalize_formula(ref)
    norm_predictions.append(list(pred_norm))
    norm_references.append([list(ref_norm)])

    if pred==ref:
        correct+=1
    if pred_norm==ref_norm:
        correct_norm+=1

# metric calculation
bleu_tok=corpus_bleu(
    references,
    predictions,
    smoothing_function=SmoothingFunction().method1
)

bleu_norm=corpus_bleu(
    norm_references,
    norm_predictions,
    smoothing_function=SmoothingFunction().method1
)

exact=correct/len(predictions)
exact_norm=correct_norm/len(predictions)

print("="*50)
print("Samples:",len(predictions))
print(f"BLEU (tok): {bleu_tok*100:.2f}")
print(f"BLEU (norm): {bleu_norm*100:.2f}")
print(f"Exact match: {exact*100:.2f}%")
print(f"Exact match (-ws): {exact_norm*100:.2f}%")