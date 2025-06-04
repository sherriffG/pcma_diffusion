import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Conv_down(nn.Module):
    def __init__(self, in_channels=256, kernel_size=5):
        super(Conv_down, self).__init__()
        layers = []
        current_in = in_channels
        for _ in range(3):
            # 步长1的卷积（特征提取）
            layers.extend([
                nn.Conv1d(current_in, 256, kernel_size, stride=1, padding=(kernel_size-1)//2),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True)
            ])
            # 步长2的卷积（下采样）
            layers.extend([
                nn.Conv1d(256, 256, kernel_size, stride=2, padding=(kernel_size-1)//2),
            ])
            current_in = 256  # 后续输入通道固定为256
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
# 编码器
class Encoder(nn.Module):
    def __init__(self,input_channels=2):
        super(Encoder, self).__init__()
        self.conv_short = Conv_down(input_channels, kernel_size=5)
        self.conv_middle = Conv_down(input_channels, kernel_size=15)
        self.conv_long = Conv_down(input_channels, kernel_size=25)

    def forward(self, x):
        short = self.conv_short(x)
        middle = self.conv_middle(x)
        long = self.conv_long(x)

        return torch.cat((short, middle, long), dim=1)
class Encoder_SNR(nn.Module):
    def __init__(self,input_channels=3):
        super(Encoder_SNR, self).__init__()
        self.conv_short = Conv_down(input_channels, kernel_size=5)
        self.conv_middle = Conv_down(input_channels, kernel_size=15)
        self.conv_long = Conv_down(input_channels, kernel_size=25)

    def forward(self, x):
        short = self.conv_short(x)
        middle = self.conv_middle(x)
        long = self.conv_long(x)

        return torch.cat((short, middle, long), dim=1)
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class Conv1d_SE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(Conv1d_SE, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=(kernel_size-1)*dilation//2)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        return x

class Separator(nn.Module):
    def __init__(self, input_channels):
        super(Separator, self).__init__()
        self.gn = nn.GroupNorm(num_groups=8, num_channels=input_channels)
        self.bottleneck1 = nn.Conv1d(input_channels, 128, 1)
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(*[Conv1d_SE(128, 128, 3, 2**i) for i in range(3)])
            for _ in range(5)
        ])
        self.bottleneck2 = nn.Conv1d(128*5, input_channels*4, 1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.gn(x)
        x = self.bottleneck1(x)
        outputs = []
        for dilated_conv in self.dilated_convs:
            output = dilated_conv(x)
            outputs.append(output)
        x = torch.cat(outputs, dim=1)
        x = self.prelu(x)
        x = self.bottleneck2(x)
        mask = torch.sigmoid(x)
        mask_chunk = torch.chunk(mask,4,dim=1)
        return mask_chunk
class Separator1(nn.Module):    #输出一路信号
    def __init__(self, input_channels):
        super(Separator1, self).__init__()
        self.gn = nn.GroupNorm(num_groups=8, num_channels=input_channels)
        self.bottleneck1 = nn.Conv1d(input_channels, 128, 1)
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(*[Conv1d_SE(128, 128, 3, 2**i) for i in range(3)])
            for _ in range(5)
        ])
        self.bottleneck2 = nn.Conv1d(128*5, input_channels*2, 1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.gn(x)
        x = self.bottleneck1(x)
        outputs = []
        for dilated_conv in self.dilated_convs:
            output = dilated_conv(x)
            outputs.append(output)
        x = torch.cat(outputs, dim=1)
        x = self.prelu(x)
        x = self.bottleneck2(x)
        mask = torch.sigmoid(x)
        mask_chunk = torch.chunk(mask,2,dim=1)
        return mask_chunk
class Conv_up(nn.Module):
    def __init__(self, in_channels=256, kernel_size=5):
        super(Conv_up, self).__init__()
        layers = []
        current_in = in_channels
        self.up1=nn.ConvTranspose1d(in_channels, 256, kernel_size, stride=2, padding=(kernel_size-1)//2,output_padding=1)
        self.bn1=nn.BatchNorm1d(256)
        self.relu1=nn.ReLU(inplace=True)
        self.conv1=nn.Conv1d(256, 256, kernel_size, stride=1, padding=(kernel_size-1)//2)

        self.up2=nn.ConvTranspose1d(in_channels, 256, kernel_size, stride=2, padding=(kernel_size-1)//2,output_padding=1)
        self.bn2=nn.BatchNorm1d(256)
        self.relu2=nn.ReLU(inplace=True)
        self.conv2=nn.Conv1d(256, 256, kernel_size, stride=1, padding=(kernel_size-1)//2)

        self.up3=nn.ConvTranspose1d(in_channels, 256, kernel_size, stride=2, padding=(kernel_size-1)//2,output_padding=1)
        self.bn3=nn.BatchNorm1d(256)
        self.relu3=nn.ReLU(inplace=True)
        self.conv3=nn.Conv1d(256, 1, kernel_size, stride=1, padding=(kernel_size-1)//2)
    def forward(self, x):
        x = self.up1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.up2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)

        x = self.up3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv3(x)
        return x
        
class Decoder(nn.Module):
    def __init__(self, output_channels=2):
        super(Decoder, self).__init__()
        # 三个上采样模块，分别对应不同的卷积核大小
        self.conv_short_up = Conv_up(256, kernel_size=5)
        self.conv_middle_up = Conv_up(256, kernel_size=15)
        self.conv_long_up = Conv_up(256, kernel_size=25)

    def forward(self, x):
        # 假设x是编码后的特征图与掩码相乘后的结果，形状为[1, 768, 100]
        # 分开处理三个特征图
        short_up = self.conv_short_up(x[:, :256, :])  # [1, 256, 800]
        middle_up = self.conv_middle_up(x[:, 256:512, :])  # [1, 256, 800]
        long_up = self.conv_long_up(x[:, 512:768, :])  # [1, 256, 800]
        output = (short_up + middle_up + long_up)/3
        return output

# 最终模型（修改forward逻辑）
class SignalSeparator(nn.Module):
    def __init__(self):
        super(SignalSeparator, self).__init__()
        self.encoder = Encoder().to(device)
        self.separator = Separator(input_channels=256*3).to(device)
        self.decoder = Decoder().to(device)
    def forward(self, x):
        # 编码阶段
        encoded = self.encoder(x)  
        
        # 分离阶段（获取两路mask）
        mask_chunk = self.separator(encoded) 
        # 解码阶段（分别处理两路信号）
        output = []
        for i, mask in enumerate(mask_chunk):
            output.append(self.decoder(mask*encoded))
        
        # 合并输出 [batch, 4 seq_len]
        return output
class SignalSeparator_SNR(nn.Module):
    def __init__(self):
        super(SignalSeparator_SNR, self).__init__()
        self.encoder = Encoder_SNR().to(device)
        self.separator = Separator(input_channels=256*3).to(device)
        self.decoder = Decoder().to(device)
    def forward(self, x):
        # 编码阶段
        encoded = self.encoder(x)  
        
        # 分离阶段（获取两路mask）
        mask_chunk = self.separator(encoded) 
        # 解码阶段（分别处理两路信号）
        output = []
        for i, mask in enumerate(mask_chunk):
            output.append(self.decoder(mask*encoded))
        
        # 合并输出 [batch, 4 seq_len]
        return output
# 最终模型（修改forward逻辑）
class SignalSeparator1(nn.Module):
    def __init__(self):
        super(SignalSeparator1, self).__init__()
        self.encoder = Encoder().to(device)
        self.separator = Separator1(input_channels=256*3).to(device)
        self.decoder = Decoder().to(device)
    def forward(self, x):
        # 编码阶段
        encoded = self.encoder(x)  
        
        # 分离阶段（获取两路mask）
        mask_chunk = self.separator(encoded) 
        # 解码阶段（分别处理两路信号）
        output = []
        for i, mask in enumerate(mask_chunk):
            output.append(self.decoder(mask*encoded))
        
        # 合并输出 [batch, 2, seq_len]
        return output
class SNREstimator(nn.Module):
    def __init__(self, input_length=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x): # (batch, 1, seq_len)
        x = self.conv_layers(x).squeeze(-1)
        return self.fc(x)
