from load_dataset import SignalDataset
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from model_diffusion import SignalSeparator
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

def prepare_dataloaders(dataset_path, mode='random', batch_size=64, train_ratio=0.9):
    """准备训练和验证数据加载器"""
    loaded_data = torch.load(f'{dataset_path}/dataset_{mode}.pt')
    dataset = SignalDataset(loaded_data)
    
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(2025)
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    return train_loader, val_loader

def initialize_model(device,lr=5e-4):
    """初始化模型和优化器"""
    model = SignalSeparator().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, optimizer

def process_batch(batch, device):
    """处理批次数据并转移到设备"""
    mixsignal_real = batch['mixsignal_real'].to(device).unsqueeze(1)
    mixsignal_imag = batch['mixsignal_imag'].to(device).unsqueeze(1)
    
    rfsignal1_real = batch['rfsignal1_real'].to(device).unsqueeze(1)
    rfsignal1_imag = batch['rfsignal1_imag'].to(device).unsqueeze(1)
    rfsignal2_real = batch['rfsignal2_real'].to(device).unsqueeze(1)
    rfsignal2_imag = batch['rfsignal2_imag'].to(device).unsqueeze(1)
    
    # snr = batch['params'][0].to(device)
    # snr_seq = snr.unsqueeze(1).unsqueeze(2).expand(-1, -1, mixsignal_imag.size(2)).to(mixsignal_imag.dtype)
    
    return {
        'mixsignal': torch.cat([mixsignal_real, mixsignal_imag], dim=1),
        'rfsignal1': torch.cat([rfsignal1_real, rfsignal1_imag], dim=1),
        'rfsignal2': torch.cat([rfsignal2_real, rfsignal2_imag], dim=1),
        'mixsignal_real': mixsignal_real,
        'mixsignal_imag': mixsignal_imag
    }

def calculate_loss(output, batch_data):
    """计算复合损失"""
    pred_rfsignal1 = torch.cat([output[0], output[1]], dim=1)
    pred_rfsignal2 = torch.cat([output[2], output[3]], dim=1)
    
    # 标准化MSE损失
    loss1 = torch.nn.functional.mse_loss(pred_rfsignal1, batch_data['rfsignal1'], reduction='none')
    loss1 = (loss1.mean(dim=[1,2]) / torch.norm(batch_data['rfsignal1'], dim=[1,2])).mean()
    
    loss2 = torch.nn.functional.mse_loss(pred_rfsignal2, batch_data['rfsignal2'], reduction='none')
    loss2 = (loss2.mean(dim=[1,2]) / torch.norm(batch_data['rfsignal2'], dim=[1,2])).mean()
    
    return loss1 + loss2

def get_diffusion_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """线性beta调度"""
    return torch.linspace(beta_start, beta_end, T)

def q_sample(x_0, t, noise, alphas_cumprod):
    """
    x_0: [batch, channel, length] 原始混合信号
    t: [batch] 当前步数
    noise: [batch, channel, length] 随机噪声
    alphas_cumprod: [T] 累积alpha
    """
    # 获取每个样本的alpha_bar
    device = x_0.device
    batch_size = x_0.size(0)
    alpha_bar = alphas_cumprod[t].reshape(batch_size, 1, 1)
    return torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise


def train_epoch(model, train_loader, optimizer, device, T, alpha_cumprod):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc='Training')
    for batch in progress_bar:
        batch_data = process_batch(batch, device)
        optimizer.zero_grad()
        
        # 原始混合含噪信号
        x_0 = torch.cat([batch_data['rfsignal1'], batch_data['rfsignal2']], dim=1)  # [batch, 4, L]
        mixsignal = batch_data['mixsignal']  # [batch, 2, L]
        batch_size = x_0.size(0)

        # 随机采样时间步t
        t = torch.randint(0, T, (batch_size,), device=device).long()

        # 计算噪声
        noise = torch.randn_like(x_0)

        # 加噪
        x_t = q_sample(x_0, t, noise, alpha_cumprod)

        # 前向传播
        output = model(x_t, t, mixsignal)
        output = torch.cat(output, dim=1)

        loss = torch.nn.functional.mse_loss(output, noise)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    return total_loss / len(train_loader)

def validate_epoch(model, val_loader, device, T, alphas_cumprod):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch_data = process_batch(batch, device)
            x_0 = torch.cat([batch_data['rfsignal1'], batch_data['rfsignal2']], dim=1)  # [batch, 4, L]
            mixsignal = batch_data['mixsignal']  # [batch, 2, L]
            batch_size = x_0.size(0)
            t = torch.randint(0, T, (batch_size,), device=device).long()
            noise = torch.randn_like(x_0)
            x_t = q_sample(x_0, t, noise, alphas_cumprod)
            output = model(x_t, t, mixsignal)
            output = torch.cat(output, dim=1)
            loss = torch.nn.functional.mse_loss(output, noise)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def plot_and_save_loss(train_losses, val_losses, save_path):
    """绘制并保存损失曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def main():
    # 配置参数
    mode_name = 'random_small'
    config = {
        'dataset_path': '/nas/datasets/yixin/PCMA/py_dataset',
        'mode': mode_name,
        'batch_size': 64,
        'num_epochs': 1000,
        'save_path': './src/pics/loss_diffusion_'+mode_name+'.pdf',
        'model_save_path': './src/models/diffusion_'+mode_name+'.pth',
        'initial_learning_rate': 5e-4
    }
    selected_cuda = "0"
    # 设备设置
    device = torch.device(f'cuda:{selected_cuda}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    torch.cuda.set_device(int(selected_cuda))  # 设置默认设备
    # 初始化数据加载器
    train_loader, val_loader = prepare_dataloaders(
        config['dataset_path'],
        mode=config['mode'],
        batch_size=config['batch_size']
    )
    
    # 初始化模型和优化器
    model, optimizer = initialize_model(device,lr=config['initial_learning_rate'])
    
    # 训练循环
    train_losses = []
    val_losses = []

    # 获取beta调度
    T = 1000  # 时间步数
    beta_schedule = get_diffusion_beta_schedule(T).to(device)
    alphas = 1. - beta_schedule
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    for epoch in range(config['num_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, device, T, alphas_cumprod)
        val_loss = validate_epoch(model, val_loader, device, T, alphas_cumprod)
        
        print(f'Epoch {epoch+1}/{config["num_epochs"]}')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    # 保存结果
    plot_and_save_loss(train_losses, val_losses, config['save_path'])
    torch.save(model.state_dict(), config['model_save_path'])
    print(f'Training completed. Model saved to {config["model_save_path"]}')

if __name__ == '__main__':
    main()