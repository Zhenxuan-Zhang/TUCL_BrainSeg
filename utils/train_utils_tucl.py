import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import numpy as np
import random
from utils.metrics_utils import cal_dice
from utils.process_utils import process_selection, generate_text_prompt

def compute_edge_mask(tensor):
    """
    Compute the boundary mask of the tensor, utilising the Sobel filter to detect gradient boundaries.
    :param tensor: Mask or predicted values of shape [B, C, D, H, W]
    :return: Boundary mask of shape [B, C, D, H, W]
    """
    kernel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]])
    kernel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]])
    kernel_x, kernel_y = kernel_x.to(tensor.device), kernel_y.to(tensor.device)
    
    edge_x = F.conv3d(tensor.float(), kernel_x.expand(tensor.shape[1], -1, -1, -1, -1), padding=1, groups=tensor.shape[1])
    edge_y = F.conv3d(tensor.float(), kernel_y.expand(tensor.shape[1], -1, -1, -1, -1), padding=1, groups=tensor.shape[1])
    
    edge_map = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    return (edge_map > 0).float() 
    
device = torch.device('cuda:0')
model_bert = BertModel.from_pretrained('bert-base-uncased').to(device)

def compute_prompt_consistency_loss(mean_pred, prompt):
    """
    Compute prompt consistency loss based on a binary prompt vector.
    The assumption is that the model output only has nonzero values in channels where prompt is 1.
    
    :param mean_pred: Tensor of shape [B, C, D, H, W] representing segmentation prediction.
    :param prompt: A list or 1D tensor of length C with binary values (e.g., [0, 1, 1]).
                   It indicates which channels should have significant output.
    :return: A scalar tensor representing the prompt consistency loss.
    """
    B, C, D, H, W = mean_pred.shape
    # Global average pooling over depth and spatial dimensions -> shape [B, C]
    global_pred = mean_pred.mean(dim=[2, 3, 4])
    
    # Convert prompt to tensor if needed and expand to match batch size
    if isinstance(prompt, list):
        prompt = torch.tensor(prompt, device=mean_pred.device, dtype=global_pred.dtype)
    if prompt.ndim == 1:
        prompt = prompt.unsqueeze(0)
    if prompt.shape[0] != B:
        prompt = prompt.expand(B, -1)
    
    # Use MSE loss: for channels with prompt==0, output should be near 0; for prompt==1, near 1.
    loss = F.mse_loss(global_pred, prompt.float())
    return loss

def train_loop(model, optimizer, scheduler, criterion, train_loader, device, epoch):
    model.train()
    running_loss = 0
    dice1_train = 0
    dice2_train = 0
    dice3_train = 0
    pbar = tqdm(train_loader)

    for it, (image, mask) in enumerate(pbar):
        # update learning rate according to the schedule
        it = len(train_loader) * epoch + it 
        
        image = image[:,:,10:138, 50:178,50:178]
        mask  = mask[:,:,10:138, 50:178,50:178]

        image, mask = image.to(device), mask.to(device)
        output = model(image)
        loss0 = criterion(output[:,0], mask[:,0])
        loss1 = criterion(output[:,1], mask[:,1])
        loss2 = criterion(output[:,2], mask[:,2])
        loss = 0.6*loss1+0.4*loss2+0.5*loss0
        

        running_loss += loss.item() 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dice1, dice2, dice3 = cal_dice(output, mask)

        dice1_train += dice1
        dice2_train += dice2
        dice3_train += dice3
        pbar.desc = "loss:{:.3f} dice1:{:.3f} dice2:{:.3f} dice3:{:.3f} ".format(loss,dice1,dice2,dice3)

    loss = running_loss / len(train_loader)
    dice1 = dice1_train / len(train_loader)
    dice2 = dice2_train / len(train_loader)
    dice3 = dice3_train / len(train_loader)
    return {'loss': loss, 'dice1': dice1, 'dice2': dice2, 'dice3': dice3}


def train_loop_sp(model, optimizer, scheduler, criterion, train_loader, device, epoch):
    model.train()
    running_loss_list, dice1_batch_list, dice2_batch_list, dice3_batch_list = [], [], [], []
    pbar = tqdm(train_loader)
    num_random = 10

    for it, (image, mask) in enumerate(pbar):
        image, mask = image[:, :, 10:138, 50:178, 50:178], mask[:, :, 10:138, 50:178, 50:178]
        image, mask = image.to(device), mask.to(device)
        
        optimizer.zero_grad()
        loss_list, dice1_list, dice2_list, dice3_list = [], [], [], []

        for i in range(num_random):
            input_list = [1, 1, 1, 1]
            idx_in = random.randint(0, 3)
            input_list[idx_in] = 0

            output_list = [1, 1, 1]
            idx_out = random.randint(0, 2)
            output_list[idx_out] = 0

            if i == 4 or i == 9:
                input_list, output_list = [1, 1, 1, 1], [1, 1, 1]

            selection = input_list + output_list

            image_processed, mask_processed = process_selection(image, mask, selection)
            text_prompt = generate_text_prompt(selection)

            output = model(image_processed, text_prompt=text_prompt)

            loss = 0.0
            if selection[4] == 1:
                loss += 0.8 * criterion(output[:, 0], mask_processed[:, 0])
            if selection[5] == 1:
                loss += 0.4 * criterion(output[:, 1], mask_processed[:, 1])
            if selection[6] == 1:
                loss += 0.5 * criterion(output[:, 2], mask_processed[:, 2])

            if loss.item() > 0:
                loss.backward()
                loss_list.append(loss.item())

                dice1, dice2, dice3 = cal_dice(output, mask_processed)
                if selection[4] == 1:
                    dice1_list.append(dice1)
                if selection[5] == 1:
                    dice2_list.append(dice2)
                if selection[6] == 1:
                    dice3_list.append(dice3)

        optimizer.step()

        running_loss_list.append(sum(loss_list) / len(loss_list) if loss_list else 0.0)
        dice1_batch_list.append(sum(dice1_list) / len(dice1_list) if dice1_list else 0.0)
        dice2_batch_list.append(sum(dice2_list) / len(dice2_list) if dice2_list else 0.0)
        dice3_batch_list.append(sum(dice3_list) / len(dice3_list) if dice3_list else 0.0)

        pbar.desc = "loss:{:.3f} dice1:{:.3f} dice2:{:.3f} dice3:{:.3f}".format(
            running_loss_list[-1], dice1_batch_list[-1], dice2_batch_list[-1], dice3_batch_list[-1]
        )

    return {
        'loss': sum(running_loss_list) / len(running_loss_list),
        'dice1': sum(dice1_batch_list) / len(dice1_batch_list),
        'dice2': sum(dice2_batch_list) / len(dice2_batch_list),
        'dice3': sum(dice3_batch_list) / len(dice3_batch_list)
    }

def train_loop_sp_uc(
    model, optimizer, scheduler, criterion, train_loader, device, epoch,
    num_forward_passes=4, uncertainty_mode='variance', alpha=1.0, beta=0.5
):
    """
    Incorporate uncertainty estimation, boundary loss, and consistency regularization.
    :param beta: Weighting coefficient for the boundary loss.
    """
    model.train()
    running_loss_list, dice1_batch_list, dice2_batch_list, dice3_batch_list = [], [], [], []
    
    pbar = tqdm(train_loader)
    num_random = 10  # Perform multiple random selections for each batch

    for it, (image, mask) in enumerate(pbar):
        image, mask = image.to(device), mask.to(device)
        image = image[:, :, 10:138, 50:178, 50:178]
        mask  = mask[:, :, 10:138, 50:178, 50:178]
        image, mask = image.to(device), mask.to(device)
        optimizer.zero_grad()
        
        loss_list, dice1_list, dice2_list, dice3_list = [], [], [], []

        for i in range(num_random):
            input_list = [1, 1, 1, 1]
            output_list = [1, 1, 1]
            selection = input_list + output_list  # Selection vector of length 7

            # Process inputs
            image_processed, mask_processed = process_selection(image, mask, selection)
            text_prompt = generate_text_prompt(selection, device)
            
            # MC Dropout sampling
            predictions = [model(image_processed, text_prompt=text_prompt).unsqueeze(0) for _ in range(num_forward_passes)]
            predictions = torch.cat(predictions, dim=0)  # Shape: [num_forward_passes, B, C, D, H, W]
            mean_pred = predictions.mean(dim=0)
            
            # Calculate uncertainty
            if uncertainty_mode == 'variance':
                uncertainty_map = predictions.var(dim=0).mean(dim=1, keepdim=True)
            else:
                prob_pred = F.softmax(mean_pred, dim=1)
                entropy_map = - (prob_pred * torch.log(prob_pred + 1e-8)).sum(dim=1, keepdim=True)
                uncertainty_map = entropy_map
            
            # Compute boundary mask
            edge_mask = compute_edge_mask(mask_processed)
            
            # Compute segmentation loss
            loss_seg = sum([criterion(mean_pred[:, i], mask_processed[:, i]) for i in range(3)])
            
            # Boundary loss: only compute segmentation loss for the boundary region
            loss_edge = sum([(criterion(mean_pred[:, i] * edge_mask[:, i], mask_processed[:, i] * edge_mask[:, i])) for i in range(3)])
            
            # Consistency Regularization (enforcing consistency among MC Dropout predictions)
            consistency_loss = ((predictions - mean_pred.unsqueeze(0)) ** 2).mean()

            prompt_loss = compute_prompt_consistency_loss(predictions, output_list)
            # Total loss computation
            loss = loss_seg * (1 + alpha * uncertainty_map.mean()) + beta * loss_edge + alpha * consistency_loss
            
            if loss.item() > 0:
                loss.backward()
                loss_list.append(loss.item())
                
                dice1, dice2, dice3 = cal_dice(mean_pred, mask_processed)
                dice1_list.append(dice1)
                dice2_list.append(dice2)
                dice3_list.append(dice3)

        optimizer.step()
        
        avg_loss = sum(loss_list) / len(loss_list) if loss_list else 0.0
        avg_dice1 = sum(dice1_list) / len(dice1_list) if dice1_list else 0.0
        avg_dice2 = sum(dice2_list) / len(dice2_list) if dice2_list else 0.0
        avg_dice3 = sum(dice3_list) / len(dice3_list) if dice3_list else 0.0

        running_loss_list.append(avg_loss)
        dice1_batch_list.append(avg_dice1)
        dice2_batch_list.append(avg_dice2)
        dice3_batch_list.append(avg_dice3)

        pbar.desc = f"loss:{avg_loss:.3f} dice1:{avg_dice1:.3f} dice2:{avg_dice2:.3f} dice3:{avg_dice3:.3f}"
    
    return {
        'loss': sum(running_loss_list) / len(running_loss_list) if running_loss_list else 0.0,
        'dice1': sum(dice1_batch_list) / len(dice1_batch_list) if dice1_batch_list else 0.0,
        'dice2': sum(dice2_batch_list) / len(dice2_batch_list) if dice2_batch_list else 0.0,
        'dice3': sum(dice3_batch_list) / len(dice3_batch_list) if dice3_batch_list else 0.0
    }

def val_loop(model, criterion, val_loader, device):
    model.eval()
    # model.train()
    running_loss = 0
    dice1_val = []
    dice2_val = []
    dice3_val = []
    pbar = tqdm(val_loader)
    with torch.no_grad():
        for image, mask in pbar:
            image = image[:,:,10:138, 50:178,50:178]
            mask  = mask[:,:,10:138, 50:178,50:178]

            mask = (mask > 0.5).float()
            image, mask = image.to(device), mask.to(device)
            
            input_list = [1, 1, 1, 1]
            
            output_list = [1, 1, 1]
            
            selection = input_list + output_list  

            image_processed, mask_processed = process_selection(image, mask, selection)
            text_prompt = generate_text_prompt(selection)
            
            output = model(image_processed, text_prompt=text_prompt)

            dice1, dice2, dice3 = cal_dice(output, mask)

            dice1_val.append(dice1.item())
            dice2_val.append(dice2.item())
            dice3_val.append(dice3.item())

            pbar.desc = "dice1:{:.3f} dice2:{:.3f} dice3:{:.3f} ".format(dice1,dice2,dice3)

    dice1 = np.mean(np.array(dice1_val))
    dice2 = np.mean(np.array(dice2_val))
    dice3 = np.mean(np.array(dice3_val))
    return {'dice1': dice1, 'dice2': dice2, 'dice3': dice3}