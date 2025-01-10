import torch
import tqdm


def Train(Train_dataloader,Valid_dataloader,EPOCH,Initial_epoch,Model,optimizer,loss_fn,devices,save_checkpoint_fn,checkpoint_path,Tensorboard):
    
    for epoch in range(Initial_epoch,EPOCH):
        
        progress_bar=tqdm.tqdm(range(len(Train_dataloader)),"Training Progress")
        
        total_values_train=0
        correct_values_train=0
        loss_values_train=0
        
        for batch_train,(input_data_train,output_data_train) in enumerate(Train_dataloader):
            
            input_encoder_train=input_data_train.to(devices)
            
            output_decoder_train=output_data_train.to(devices)
            
            initial_input_decoder=torch.zeros_like(output_data_train).to(devices)
            initial_input_decoder[:,0]=output_data_train[:,0] # Add start token to input decoder
            
            output_decoder_train=output_decoder_train[:,1:] # To remove start token from out decoder
            optimizer.zero_grad()
            output_train=Model(input_encoder_train,initial_input_decoder,output_decoder_train)
            # We Clip target data as output transformer because we stopped model when we reached stop token but target values is padded max len so we need to clip.
            output_decoder_train=output_decoder_train[:,:output_train.shape[1]]
            
            loss_train=loss_fn(output_train.reshape(-1,output_train.shape[-1]),output_decoder_train.reshape(-1))
            loss_train.backward()
            optimizer.step()
            
            
            _,pred_train=torch.max(output_train,-1)
            correct_values_train+=(pred_train==output_decoder_train).sum().item()
            total_values_train+=(output_decoder_train.size(0)*output_decoder_train.size(1))
            loss_values_train+=loss_train.item()
            
            progress_bar.update(1)
            
            if batch_train %40==0 and batch_train>0:
                with torch.no_grad():
                    
                    total_values_valid=0
                    correct_values_valid=0
                    loss_values_valid=0
                    
                    for batch_valid,(input_data_valid,output_data_valid) in enumerate(Valid_dataloader):
                
                        input_encoder_valid=input_data_valid.to(devices)
                        output_decoder_valid=output_data_valid.to(devices)
                        
                        output_decoder_valid=output_decoder_valid[:,1:] # To remove start token from out decoder
                        
                        output_valid=Model(input_encoder_valid,initial_input_decoder,output_decoder_valid)
                        
                        output_decoder_valid=output_decoder_valid[:,:output_valid.shape[1]]
                        loss_valid=loss_fn(output_valid.reshape(-1,output_valid.shape[-1]),output_decoder_valid.reshape(-1))
                        
                        _,pred_valid=torch.max(output_valid,-1)
                        correct_values_valid+=(pred_valid==output_decoder_valid).sum().item()
                        total_values_valid+=(output_decoder_valid.size(0)*output_decoder_valid.size(1))
                        loss_values_valid+=loss_valid.item()           
                
                progress_bar.set_postfix({"EPOCH":epoch,
                                         "Batch":batch_train+1,
                                         "Acc_Train":(correct_values_train/total_values_train)*100,
                                         "Loss_Train":(loss_values_train/(batch_train+1)),
                                         "Acc_Valid":(correct_values_valid/total_values_valid)*100,
                                         "Loss_Valid":(loss_values_valid/(batch_valid+1))})

        # Tensorboard
        Tensorboard.add_scalar("Accuracy Train",(correct_values_train/total_values_train)*100,global_step=epoch)
        Tensorboard.add_scalar("Loss Train",(loss_values_train/(batch_train+1)),global_step=epoch)
        Tensorboard.add_scalar("Accuracy Valid",(correct_values_valid/total_values_valid)*100,global_step=epoch)
        Tensorboard.add_scalar("Loss Valid",(loss_values_valid/(batch_valid+1)),global_step=epoch)
        
            
        save_checkpoint_fn(epoch,optimizer,Model,checkpoint_path)
    

            
            
            
            
        
    
    
    