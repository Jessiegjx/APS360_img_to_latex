import matplotlib.pyplot as plt


# ==========================
# Results from training
# ==========================

epochs = list(range(1, 11))
# Loss
train_loss = [
    3.0489,
    2.5298,
    2.4037,
    2.3326,
    2.2865,
    2.2479,
    2.2181,
    2.1974,
    2.1738,
    2.1560
]

val_loss = [
    2.6329,
    2.4653,
    2.3715,
    2.3254,
    2.2906,
    2.2768,
    2.2635,
    2.2412,
    2.2238,
    2.2155
]


# Accuracy
train_token_acc = [
    0.385,
    0.454,
    0.474,
    0.484,
    0.491,
    0.497,
    0.502,
    0.504,
    0.508,
    0.511
]

val_token_acc = [
    0.436,
    0.465,
    0.479,
    0.485,
    0.492,
    0.491,
    0.494,
    0.497,
    0.501,
    0.502
]


train_exact = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
]

val_exact= [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
]




# ==========================
# Loss plot
# ==========================

plt.figure(figsize=(6,4))

plt.plot(
    epochs,
    train_loss,
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.title("Baseline CNN-RNN Training and Validation Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "baseline_loss_curve.png",
    dpi=300
)

plt.show()



# ==========================
# Accuracy plot
# ==========================

plt.figure(figsize=(6,4))


plt.plot(
    epochs,
    train_token_acc,
    label="Training Token Accuracy"
)

plt.plot(
    epochs,
    val_token_acc,
    label="Validation Token Accuracy"
)


plt.plot(
    epochs,
    train_exact,
    label="Training Exact Match"
)

plt.plot(
    epochs,
    val_exact,
    label="Validation Exact Match"
)


plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Baseline CNN-RNN Token Accuracy and Exact Match")


plt.ylim(0, 0.6)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "baseline_accuracy_curve.png",
    dpi=300
)

plt.show()