import re
import matplotlib.pyplot as plt

IM2LATEX_LOG = r"D:\APS360_proj\training_log_140epoch.txt.txt"
HME_LOG = r"D:\APS360_proj\training_log_last30epoch.txt.txt"

def extract_im2latex_loss(path):
    with open(path,"r") as f:
        text=f.read()
    train_loss=re.findall(
        r"Train loss:\s*([0-9.]+)",
        text
    )
    val_loss=re.findall(
        r"Validation loss:\s*([0-9.]+)",
        text
    )
    return list(map(float,train_loss)), list(map(float,val_loss))


def extract_hme_loss(path):
    with open(path,"r") as f:
        text=f.read()
    train_loss=re.findall(
        r"Train loss:\s*([0-9.]+)",
        text
    )
    val_loss=re.findall(
        r"Validation loss:\s*([0-9.]+)",
        text
    )
    return list(map(float,train_loss)), list(map(float,val_loss))


def extract_im2latex_metrics(path):
    with open(path,"r") as f:
        text=f.read()
    acc=re.findall(
        r"Token accuracy:\s*([0-9.]+)",
        text
    )
    exact=re.findall(
        r"Exact match:\s*([0-9.]+)",
        text
    )
    return list(map(float,acc)), list(map(float,exact))


def extract_hme_metrics(path):
    with open(path,"r") as f:
        text=f.read()
    acc=re.findall(
        r"Validation token accuracy:\s*([0-9.]+)",
        text
    )
    exact=re.findall(
        r"Validation exact match:\s*([0-9.]+)",
        text
    )

    return list(map(float,acc)), list(map(float,exact))


# -------------------------
# Extract separately
# -------------------------

im_train_loss,im_val_loss=extract_im2latex_loss(IM2LATEX_LOG)

# remove duplicated HME-style loss entries accidentally captured
im_train_loss = im_train_loss[:-31]
im_val_loss = im_val_loss[:-31]

hme_train_loss, hme_val_loss = extract_hme_loss(HME_LOG)

im_acc,im_exact=extract_im2latex_metrics(IM2LATEX_LOG)
hme_acc,hme_exact=extract_hme_metrics(HME_LOG)


print("IM2LaTeX loss:",len(im_train_loss))
print("HME loss:",len(hme_train_loss))
print("IM2LaTeX metrics:",len(im_acc))
print("HME metrics:",len(hme_acc))


# -------------------------
# Combine
# -------------------------

train_loss=im_train_loss+hame_train_loss if False else im_train_loss+hme_train_loss
val_loss=im_val_loss+hme_val_loss

val_acc=im_acc+hme_acc
val_exact=im_exact+hme_exact


# -------------------------
# Loss plot
# -------------------------

epochs_loss=range(1,len(train_loss)+1)

plt.figure(figsize=(8,5))

plt.plot(
    epochs_loss,
    train_loss,
    label="Training Loss"
)

plt.plot(
    epochs_loss,
    val_loss,
    label="Validation Loss"
)

plt.axvline(
    140,
    linestyle="--",
    label="HME100K Fine-tuning"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("loss_curve.png",dpi=300)
plt.show()


# -------------------------
# Validation accuracy plot
# -------------------------

epochs_metric=range(1,len(val_acc)+1)

plt.figure(figsize=(8,5))

plt.plot(
    epochs_metric,
    val_acc,
    label="Validation Token Accuracy"
)

plt.plot(
    epochs_metric,
    val_exact,
    label="Validation Exact Match"
)

plt.axvline(
    140,
    linestyle="--",
    label="HME100K Fine-tuning"
)

plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Validation Token Accuracy and Exact Match")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("validation_metrics.png",dpi=300)
plt.show()