import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self,seq_hidden_dim, hidden_dim_nc ,seq_hidden_dim,device='cpu'):
        super(SelfAttentiveEncoder, self).__init__()
        self.W1 = nn.Parameter(nn.init.xavier_uniform_(torch.empty((seq_hidden_dim, hidden_dim_nc))).to(self.device))
        self.W2 = nn.Parameter(nn.init.xavier_uniform_(torch.empty((hidden_dim_nc, n_hop))).to(self.device))



    def forward(self, H):
        size=H.size() #expected = [batch_size,19,3,512]
        print("Size:")
        print(size)
        x=nn.tanh(nn.bmm(H.view(size[0],size[1]*size[2]),self.W1))
        x=nn.bmm(self.W2,x)
        A=nn.Softmax(x,dim=0)
        E=nn.bmm(torch.transpose(A, 1, 2),H)
        
        return E

       

""" 
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hops, seq_len]

        alphas = self.softmax(alphas.view(-1, size[1]))  # [bsz*hop, seq_len]

        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, seq_len]

        return torch.bmm(alphas, outh), alphas """

class VGGM(pl.LightningModule):
    
    def __init__(self, n_classes=1251):
        super(VGGM, self).__init__()
        self.n_classes=n_classes
        self.conv_part=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1)),
            ('bn1', nn.BatchNorm2d(64, momentum=0.1)),
            ('relu1', nn.ReLU()),
            ('mpool1', nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=1)),
            ('bn2', nn.BatchNorm2d(128, momentum=0.5)),
            ('relu2', nn.ReLU()),
            ('mpool2', nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))),
            ('conv3', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1)),
            ('bn3', nn.BatchNorm2d(256, momentum=0.5)),
            ('relu3', nn.ReLU()),
            ('mpool3', nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))),
            ('conv4', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=1)),
            ('bn4', nn.BatchNorm2d(256, momentum=0.5)),
            ('relu4', nn.ReLU()),
            ('mpool4', nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))),
            ('conv5', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=1)),
            ('bn5', nn.BatchNorm2d(512, momentum=0.5)),
            ('relu5', nn.ReLU()),
            ('mpool5', nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))),
            ]))

        self.attention=nn.TransformerEncoderLayer(d_model=self.hidden_size_lstm*2,dim_feedforward=512,nhead=self.num_heads_self_attn)

    
        self.avgpool = nn.AvgPool2d((4, 1))
        self.classifier=nn.Sequential(OrderedDict([
            ('fc7', nn.Linear(4096, 1024)),
            #('drop1', nn.Dropout()),
            ('relu7', nn.ReLU()),
            ('fc8', nn.Linear(1024, n_classes))]))
    
    def forward(self, inp):
        x=self.conv_part(inp)
        x=self.attention(x)
        x=self.avgpool(x)
        x=self.classifier(nn.Flatten(x))

        return x




  



