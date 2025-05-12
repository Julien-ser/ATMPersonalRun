import os
import torch
import torch.distributed as dist

def setup():
    # Set up master address and port (adjust MASTER_ADDR to your master's IP)
    os.environ['MASTER_ADDR'] = 'IP'  # Replace with your master machine IP
    os.environ['MASTER_PORT'] = '12355'      # Free port

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # Set the current device for the current rank

    return rank

def run():
    rank = setup()

    # Basic message to show that the rank is initialized correctly
    print(f"[Rank {rank}] Initialized")

    # Sync between ranks (simple all-reduce operation)
    tensor = torch.ones(1).cuda(rank)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Print out the result to verify communication
    print(f"[Rank {rank}] All-reduced sum: {tensor.item()}")

    # Clean up after run
    dist.destroy_process_group()

if __name__ == "__main__":
    run()
