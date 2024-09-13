import torch
import torch.nn as nn
  
class SupConLoss(nn.Module):
    def __init__(self, temperature=1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, a_features, v_features, audio_only_indices, image_only_indices, labels):

        batch_size = labels.shape[0]
        
        a_loss = 0
        v_loss = 0
        
        audio_only_features = a_features[:,audio_only_indices]
        image_audio_only_features = v_features[:,audio_only_indices]
        
        image_only_features = v_features[:,image_only_indices]
        audio_image_only_features = a_features[:,image_only_indices]
        
        for i in range(batch_size):
            v_loss += -torch.log(torch.exp(torch.matmul(image_audio_only_features[i],audio_only_features[i].detach().T)/self.temperature).sum()/torch.exp(torch.matmul(image_audio_only_features[i],audio_only_features.detach().T)/self.temperature).sum())
            
            a_loss += -torch.log(torch.exp(torch.matmul(audio_image_only_features[i],image_only_features[i].detach().T)/self.temperature).sum()/torch.exp(torch.matmul(audio_image_only_features[i],image_only_features.detach().T)/self.temperature).sum())
           
            
        
        v_loss /= batch_size
        a_loss /= batch_size
        
        loss = v_loss + a_loss

        return loss
    