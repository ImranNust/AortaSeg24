import torch
from tqdm import tqdm
import os  # Import missing module
from monai.inferers import sliding_window_inference

def validation(epoch_iterator_val, patch_size, num_samples, model, decollate_batch, post_label, post_pred, dice_metric, device, global_step):  # Include missing arguments
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
            val_outputs = sliding_window_inference(val_inputs, (patch_size, patch_size, patch_size), num_samples, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val


def save_model(model, saved_model_dir, filename="best_metric_model.pth"):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    torch.save(model.state_dict(), os.path.join(saved_model_dir, filename))

class Trainer:
    def __init__(self,
                 model,
                 loss_function,
                 optimizer,
                 max_iterations,
                 eval_num,
                 saved_model_dir,
                 device,
                 patch_size,  # Include missing argument
                 num_samples,  # Include missing argument
                 decollate_batch,  # Include missing argument
                 post_label,  # Include missing argument
                 post_pred,  # Include missing argument
                 dice_metric):  # Include missing argument
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.max_iterations = max_iterations
        self.eval_num = eval_num
        self.saved_model_dir = saved_model_dir
        self.device = device  # Added missing attribute

        self.epoch_loss_values = []
        self.metric_values = []

        self.patch_size = patch_size  # Store patch size
        self.num_samples = num_samples  # Store num samples
        self.decollate_batch = decollate_batch  # Store decollate batch function
        self.post_label = post_label  # Store post label function
        self.post_pred = post_pred  # Store post pred function
        self.dice_metric = dice_metric  # Store dice metric function
        
    

    def train(self, global_step, train_loader, val_loader, dice_val_best, global_step_best):
        self.model.train()
        epoch_loss = 0
        step = 0
        epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)

        for step, batch in enumerate(epoch_iterator):
            step += 1
            x, y = (batch["image"].to(self.device), batch["label"].to(self.device))  # Added self.device
            logit_map = self.model(x)
            loss = self.loss_function(logit_map, y)
            loss.backward()
            epoch_loss += loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, self.max_iterations, loss))

            if (global_step % self.eval_num == 0 and global_step != 0) or global_step == self.max_iterations:
                epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
                dice_val = validation(epoch_iterator_val, self.patch_size, self.num_samples, self.model, self.decollate_batch, self.post_label, self.post_pred, self.dice_metric, self.device, global_step)  # Pass missing arguments
                epoch_loss /= step
                self.epoch_loss_values.append(epoch_loss)
                self.metric_values.append(dice_val)

                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    save_model(self.model, self.saved_model_dir, "best_metric_model.pth")
                    print("Model Was Saved ! Current Best Avg. Dice: {} | Current Avg. Dice: {}".
                          format(dice_val_best, dice_val))
                else:
                    print("Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".
                          format(dice_val_best, dice_val))

            global_step += 1

        return global_step, dice_val_best, global_step_best

    
