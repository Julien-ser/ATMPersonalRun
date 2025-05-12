import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.rpc import RRef


class RemoteEmbedding(nn.Module):
    def __init__(self, num_embeddings=1000, embedding_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, input):
        return self.embedding(input)


class Trainer(nn.Module):
    def __init__(self, embedding_rref):
        super().__init__()
        self.embedding_rref = embedding_rref
        self.fc = nn.Linear(128, 10).cuda()

    def forward(self, input):
        emb = rpc.rpc_sync(self.embedding_rref.owner(), _remote_forward, args=(self.embedding_rref, input))
        output = self.fc(emb.cuda())
        return F.log_softmax(output, dim=1)


def _remote_forward(rref, input):
    return rref.local_value()(input)


def run_master():
    print("[Master] Creating RemoteEmbedding RRef...")
    embedding_rref = rpc.remote("parameter_server", RemoteEmbedding, args=())
    
    futs = []
    for rank in [2, 3]:
        trainer_name = f"trainer{rank - 2}"
        print(f"[Master] Sending async RPC to {trainer_name}")
        futs.append(
            rpc.rpc_async(trainer_name, run_trainer, args=(embedding_rref, rank))
        )

    print("[Master] Waiting for all trainers to finish")
    [f.wait() for f in futs]
    print("[Master] All trainers done")



def run_trainer(embedding_rref, rank):
    print(f"[Rank {rank}] Starting trainer setup...")
    
    torch.cuda.set_device(rank - 2)
    print(f"[Rank {rank}] CUDA device set")

    dist.init_process_group(backend="nccl", rank=rank, world_size=4)
    print(f"[Rank {rank}] Process group initialized")

    model = Trainer(embedding_rref).cuda()
    ddp = DDP(model, device_ids=[rank - 2])
    print(f"[Rank {rank}] DDP model created")

    optimizer = torch.optim.SGD(ddp.parameters(), lr=0.01)

    for epoch in range(5):
        inputs = torch.randint(0, 1000, (32,), dtype=torch.long).cuda()
        labels = torch.randint(0, 10, (32,), dtype=torch.long).cuda()

        optimizer.zero_grad()
        output = ddp(inputs)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()

        print(f"[Rank {rank}] Epoch {epoch} Loss: {loss.item()}")



def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rpc_backend = "nccl" if torch.cuda.is_available() else "gloo"
    
    print(f"[Rank {rank}] Starting init_rpc")

    rpc.init_rpc(
        name={0: "master", 1: "parameter_server", 2: "trainer0", 3: "trainer1"}[rank],
        rank=rank,
        world_size=world_size
    )

    print(f"[Rank {rank}] RPC initialized")

    if rank == 0:
        print(f"[Rank {rank}] Running master logic")
        run_master()

    print(f"[Rank {rank}] Shutting down RPC")
    rpc.shutdown()



if __name__ == "__main__":
    main()
