

import torch.multiprocessing.spawn


def DataParallel():
    """
    DP 实际上已经被弃用了，一般还是用 DDP 之类的 
    """

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    print("Data Parallel testing...")

    # 1. 定义模型和数据
    input_size = 3
    num_samples = 16
    batch_size = 4

    # 2. 生成随机数据
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, 2, (num_samples, ))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 定义一个简单的模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(input_size, 2)
        
        def forward(self, x):
            return self.fc(x)
    
    model = SimpleModel()
    device_ids = [0, 1] if torch.cuda.device_count() >= 2 else [0]
    model = nn.DataParallel(model, device_ids=device_ids)
    model.to(f'cuda:{device_ids[0]}')  # 默认第一个卡为主GPU

    # 3. 定义优化起
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 4. 训练循环
    for inputs, labels in loader:
        # 所有的输入输出一开始在主GPU上
        inputs = inputs.to(f'cuda:{device_ids[0]}')
        labels = labels.to(f'cuda:{device_ids[0]}')

        # 主GPU分发各自的数据到WorkerGPU上进行前向传播计算
        outputs = model(inputs)

        # 汇总到主 GPU 上进行全局 Loss 计算
        loss = nn.CrossEntropyLoss()(outputs, labels)

        # 每个workerGPU各自计算梯度, 然后回传到主GPU上
        optimizer.zero_grad()
        loss.backward()

        # 主GPU综合梯度进行更新，然后把更新的参数同步到workerGPU上
        optimizer.step()


def DistributedDataParallel():
    import os
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import Dataset, DataLoader
    from torch.utils.data.distributed import DistributedSampler

    # 1. 初始化分布式进程组
    def setup(rank, world_size):
        """
        rank: 进程编号
        world_size: 总进程数量 
        """

        # 多主机情况下需要设置
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    # 2. 定义简单模型和数据
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)
        def forward(self, x):
            return self.fc(x)
    
    class FakeDataset(Dataset):
        def __len__(self):
            return 100
        
        def __getitem__(self, idx): 
            x = torch.randn(10)
            y = torch.randint(0, 2, (1, ).item())
            return x, y
    
    # 3. 每个进程的执行函数
    def train(rank, world_size):
        setup(rank, world_size)

        # 3.1 创建模型并包装为 DDP
        model = SimpleModel().to(rank)
        ddp_model = DDP(model, device_ids=[rank])

        # 3.2 分布式数据加载
        dataset = FakeDataset()
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        loader = DataLoader(dataset, batch_size=16, sampler=sampler)

        # 3.3 优化器
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

        # 3.4 训练
        for x, y in loader:
            x, y = x.to(rank), y.to(rank)

            # 前向计算, loss自动通信汇总获得全局 loss
            outputs = ddp_model(x)
            loss = nn.CrossEntropyLoss()(outputs, y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度自动进行同步，不需要手动操作

            # 更新参数
            optimizer.step()
        
        # 清楚进程组
        dist.destroy_process_group()
    
    # 开始执行
    world_size = 2
    torch.multiprocessing.spawn(train, args=(world_size, ), nprocs=world_size)

def FullyShardedDataParallel():
    import os
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributed import init_process_group, destroy_process_group
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        CPUOffload,
        BackwardPrefetch,
        ShardingStrategy
    )
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    from torch.utils.data import Dataset, DataLoader
    from torch.utils.data.distributed import DistributedSampler

    class TransformerBlock(nn.Module):
        def __init__(self, hidden_dim=1024):
            super().__init__()
            self.attn = nn.MultiheadAttention(hidden_dim, 8)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, 4*hidden_dim),
                nn.GELU(),
                nn.Linear(4*hidden_dim, hidden_dim)
            )
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
        
        def forward(self, x):
            # 自注意力
            attn_output, _ = self.attn(x, x, x)
            x = self.norm1(x + attn_output)  # 残差链接

            mlp_output = self.mlp(x)
            x = self.norm2(x + mlp_output)  # 残差链接
            return x
    
    class LargeModel(nn.Module):
        def __init__(self, num_layers=12, hidden_dim=1024):
            super().__init__()
            self.layers = nn.ModuleList(
                [TransformerBlock(hidden_dim) for _ in range(num_layers)]
            )
            self.proj = nn.Linear(hidden_dim, 1000) # 分类头

        def forward(self, x):
            # 输入形状：(seq_len, batch_size, hidden_dim)
            for layer in self.layers:
                x = layer(x)
            return self.proj(x.mean(dim=0))  # 取序列平均后分类
    
    def train(rank, world_size):
        # 初始化分布环境
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

        model = LargeModel().to(rank)

        # 自定义分片策略, 参数超过 1 亿的模块独立分片
        auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=100_000_000)

        model = FSDP(
            model,
            device_id=rank,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # 全分片模式
            cpu_offload=CPUOffload(offload_params=False),  # 禁用CPU卸载
            mixed_precision=True
        )

        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

        class FakeDataset(Dataset):
            def __len__(self):
                return 1000
            def __getitem__(self, index):
                x = torch.randn(128, 1024)
                y = torch.randint(0, 1000, (1, )).item()
                return x, y
        
        dataset = FakeDataset()
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        loader = DataLoader(dataset, batch_size=16, sampler=sampler)

        # 训练
        model.train()
        for epoch in range(10):
            sampler.set_epoch(epoch)
            for x, y in loader:
                x = x.to(rank).transpose(0 ,1)
                y = y.to(rank)

                outputs = model(x)
                loss = nn.CrossEntropyLoss()(outputs, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        destroy_process_group()
    
    # 启动训练
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size, ), nprocs=world_size, join=True)

if __name__ == "__main__":
    # DataParallel()
    # DistributedDataParallel()
    FullyShardedDataParallel()