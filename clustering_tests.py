# train.py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def train(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Running basic DDP example on rank {rank}.")

    model = torch.nn.Linear(10, 1).cuda(rank)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)

    for _ in range(10):
        inputs = torch.randn(20, 10).cuda(rank)
        labels = torch.randn(20, 1).cuda(rank)

        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f"Rank {rank}, Loss: {loss.item()}")

    dist.destroy_process_group()

def run():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    train(rank, world_size)

if __name__ == "__main__":
    run()