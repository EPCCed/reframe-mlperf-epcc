import torch
import torch.distributed as dist

from ML_HPC.gc import GlobalContext

gc = GlobalContext()


def compute_score(prediction: torch.Tensor, gt: torch.Tensor, num_classes: int) -> torch.Tensor:
        # flatten input
        batch_size = gt.shape[0]
        tpt = torch.zeros((batch_size, num_classes), dtype=torch.long, device=prediction.device)
        fpt = torch.zeros((batch_size, num_classes), dtype=torch.long, device=prediction.device)
        fnt = torch.zeros((batch_size, num_classes), dtype=torch.long, device=prediction.device)
        
        # create views:
        pv = prediction.view(batch_size, -1)
        gtv = gt.view(batch_size, -1)
        
        # compute per class accuracy
        for j in range(0, num_classes):
            # compute helper tensors
            pv_eq_j = (pv == j)
            pv_ne_j = (pv != j)
            gtv_eq_j = (gtv == j)
            gtv_ne_j = (gtv != j)
            
            #true positve: prediction and gt agree and gt is of class j: (p == j) & (g == j)
            tpt[:, j] = torch.sum(torch.logical_and(pv_eq_j, gtv_eq_j), dim=1)
            
            #false positive: prediction is of class j and gt not of class j: (p == j) & (g != j)
            fpt[:, j] = torch.sum(torch.logical_and(pv_eq_j, gtv_ne_j), dim=1)

            #false negative: prediction is not of class j and gt is of class j: (p != j) & (g == j)
            fnt[:, j] = torch.sum(torch.logical_and(pv_ne_j, gtv_eq_j), dim=1)
            
        # compute IoU per batch
        uniont = (tpt + fpt + fnt) * num_classes
        iout = torch.sum(torch.nan_to_num(tpt.float() / uniont.float(), nan=1./float(num_classes)), dim=1)
            
        # average over batch dim
        iout = torch.mean(iout)
        
        return iout


def validate(net, criterion, validation_loader):
    #eval
    net.eval()

    count_sum_val = torch.zeros((1), dtype=torch.float32, device=gc.device)
    loss_sum_val = torch.zeros((1), dtype=torch.float32, device=gc.device)
    iou_sum_val = torch.zeros((1), dtype=torch.float32, device=gc.device)

    # disable gradients
    with torch.no_grad():

        # iterate over validation sample
        step_val = 0
        # only print once per eval at most
        for inputs_val, label_val, filename_val in validation_loader:

            #send to device
            inputs_val = inputs_val.to(gc.device)
            label_val = label_val.to(gc.device)
            
            # forward pass
            outputs_val = net.forward(inputs_val)
            loss_val = criterion(outputs_val, label_val)

            # accumulate loss
            loss_sum_val += loss_val
        
            #increase counter
            count_sum_val += 1.
        
            # Compute score
            predictions_val = torch.argmax(torch.softmax(outputs_val, 1), 1)
            iou_val = compute_score(predictions_val, label_val, num_classes=3)
            iou_sum_val += iou_val
        
            #increase eval step counter
            step_val += 1
                
        # average the validation loss
        if dist.is_initialized():
            dist.all_reduce(count_sum_val, op=dist.ReduceOp.SUM, async_op=False)
            dist.reduce(loss_sum_val, dst=0, op=dist.ReduceOp.SUM)
            dist.all_reduce(iou_sum_val, op=dist.ReduceOp.SUM, async_op=False)
        iou_avg_val = iou_sum_val.item() / count_sum_val.item()

    # print results

    stop_training = False
    if (iou_avg_val >= gc["training"]["target_iou"]):
        stop_training = True

    # set to train
    net.train()
    
    return stop_training