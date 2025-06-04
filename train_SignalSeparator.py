from load_dataset import SignalDataset
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from model_complex import SignalSeparator
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

def train_epoch(model, train_loader, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc='Training')
    for batch in progress_bar:
        batch_data = process_batch(batch, device)
        optimizer.zero_grad()
        output = model(batch_data['mixsignal'])
        loss = calculate_loss(output, batch_data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    return total_loss / len(train_loader)

def validate_epoch(model, val_loader, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch_data = process_batch(batch, device)
            output = model(batch_data['mixsignal'])
            loss = calculate_loss(output, batch_data)
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
    mode_name = 'qpsk_all_params_10000'
    config = {
        'dataset_path': '/nas/datasets/yixin/PCMA/py_dataset',
        'mode': mode_name,
        'batch_size': 64,
        'num_epochs': 50,
        # 'save_path': './test/loss_SigSep_'+mode_name+'.png',
        # 'model_save_path': 'signal_separator_'+mode_name+'.pth',
        'save_path': './src/py_dataset/loss_SigSep_'+mode_name+'.png',
        'model_save_path': 'signal_separator_'+mode_name+'.pth',
        'initial_learning_rate': 5e-4
    }
    selected_cuda = "1"
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
    for epoch in range(config['num_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate_epoch(model, val_loader, device)
        
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