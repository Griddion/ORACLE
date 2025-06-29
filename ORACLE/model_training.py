#%%
from minbpe import BasicTokeniser
import torch
from transformer_model import GPTLanguageModel
from typing import Tuple, Dict
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt



tokeniser = BasicTokeniser()

tokeniser.load(model_file = "output/tokeniser/my_tokeniser.model")

def get_vocab_size(tokeniser: BasicTokeniser) -> int:
    vocab = tokeniser.vocab
    special_tokens = tokeniser.special_tokens

    return len(vocab) + len(special_tokens)

torch.manual_seed(3647)

block_size = 256
n_embd = 512
n_head = 8
n_layer = 4
dropout = 0.2
batch_size = 64
vocab_size = get_vocab_size(tokeniser)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%
model = GPTLanguageModel(
    vocab_size = vocab_size,
    n_embd = n_embd,
    n_head = n_head,
    block_size = block_size,
    n_layer = n_layer,
    dropout = dropout,
    device = device
)
model = model.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

#%%
# Data prep
with open("corpus data/training_corpus.txt", "r", encoding = "utf-8") as f:
    text_sequence = f.read()

encoded_text_sequence = tokeniser.encode(text_sequence)
print(len(encoded_text_sequence))

#%%
# Split into train and test
data = torch.tensor(encoded_text_sequence, dtype = torch.long)
split_index = int(0.9 * len(data))
train_data = data[:split_index]
val_data = data[split_index:]

#%%
# Data Loader
class TextDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int) -> None:
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index: index + self.block_size]
        y = self.data[index + 1: index + self.block_size + 1]
        return x, y

def get_dataloaders(
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        block_size: int,
        batch_size: int,
        device: torch.device
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = TextDataset(train_data.to(device), block_size)
    val_dataset = TextDataset(val_data.to(device), block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        shuffle = False,
    )

    return train_loader, val_loader

#%%
train_loader, val_loader = get_dataloaders(
    train_data = train_data,
    val_data = train_data,
    block_size = block_size,
    batch_size = batch_size,
    device = device,
)
x, y = next(iter(train_loader))
print(x.shape, y.shape)

#%%
# Training
@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    eval_iters: int
) -> Dict[str, float]:
    output = {}
    model.eval()

    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        for i, (x, y) in enumerate(loader):
            if i >= eval_iters:
                break
            with torch.no_grad():
                _, loss = model(x, y)
            losses[i] = loss.item()
            output[split] = losses.mean().item()

        model.train()
        return output

#%%
def save_checkpoint(
        model: GPTLanguageModel,
        optimiser: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        file_path: str = "checkpoint.pth"
) -> None:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimiser_state_dict': optimiser.state_dict(),
        loss: loss,
    }
    torch.save(checkpoint, file_path)

# %%
max_iters = 1
eval_interval = 100
eval_iters = 200
learning_rate = 3e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
train_loader, val_loader = get_dataloaders(
    train_data=train_data,
    val_data=val_data,
    block_size=block_size,
    batch_size=batch_size,
    device=device
)

train_losses = []
val_losses = []

for iteration in range(max_iters):
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        # Evaluation
        if batch_idx % eval_interval == 0 or batch_idx == len(train_loader) - 1:
            losses = estimate_loss(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                eval_iters=min(eval_iters, len(val_loader))
            )
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])

            print(
                f"iteration {iteration} / step {batch_idx}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        # Training step
        logits, loss = model(x_batch, y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=iteration,
        loss=loss.item(),
        file_path=f"output/pre_training/checkpoint_{iteration}.pth"
    )
# %%
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss", marker='o')
plt.plot(val_losses, label="Validation Loss", marker='o')
plt.xlabel("Evaluation Step")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Time")
plt.legend()
plt.grid()
plt.show()
# %%
input_tokens = tokeniser.encode("bonjour, çava?")
input_tokens = torch.tensor(
    input_tokens, dtype=torch.long).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    output = model.generate(input_tokens=input_tokens, max_new_tokens=100)

print(tokeniser.decode(output[0].tolist()))