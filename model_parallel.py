import torch
import torch.nn as nn
import torch.optim as optim


from torchvision.models.resnet import ResNet, Bottleneck

num_classes = 1000



model = ResNet

    

class ModelParallelResNet50(model):
    def __init__(self, ranks, *args, **kwargs):
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        #super(ModelParallelResNet50, self).__init__(
        #    *args, **kwargs)
        self.ranks = ranks

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:'+str(self.ranks[0]))

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:'+str(self.ranks[1]))
        
        self.fc.to('cuda:'+str(self.ranks[1]))

    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:'+str(self.ranks[1])))
        return self.fc(x.view(x.size(0), -1))







class PipelineParallelResNet50(ModelParallelResNet50):
    def __init__(self, ranks, split_size=16, *args, **kwargs):
        super(PipelineParallelResNet50, self).__init__(ranks, *args, **kwargs)

        self.split_size = split_size
        self.ranks = ranks

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        s_prev = self.seq1(s_next).to('cuda:'+str(self.ranks[1]))
        ret = []

        for s_next in splits:
            # A. s_prev runs on cuda:1
            s_prev = self.seq2(s_prev)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

            # B. s_next runs on cuda:0, which can run concurrently with A
            s_prev = self.seq1(s_next).to('cuda:'+str(self.ranks[1]))

        s_prev = self.seq2(s_prev)
        ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

        return torch.cat(ret)





