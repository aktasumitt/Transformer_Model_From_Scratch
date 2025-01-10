import torch
import tqdm


def Test(Test_Dataloader,devices,Model,loss_fn):

    with torch.no_grad():

        total_values_test = 0
        correct_values_test = 0
        loss_values_test = 0
        progress_bar=tqdm.tqdm(range(len(Test_Dataloader)),"Test Progress")
        for batch_test, (input_data_test, output_data_test) in enumerate(Test_Dataloader):

            input_encoder_test = input_data_test.to(devices)
            output_decoder_test = output_data_test.to(devices)

            # To remove start token from out decoder
            output_decoder_test = output_decoder_test[:, 1:]
            initial_input_decoder=torch.zeros_like(output_data_test).to(devices)
            initial_input_decoder[:,0]=output_data_test[:,0]

            output_test = Model(input_encoder_test, initial_input_decoder, output_decoder_test)

            output_decoder_test = output_decoder_test[:,:output_test.shape[1]]
            loss_test = loss_fn(output_test.reshape(-1, output_test.shape[-1]), output_decoder_test.reshape(-1))

            _, pred_test = torch.max(output_test, -1)
            correct_values_test += (pred_test==output_decoder_test).sum().item()
            total_values_test += (output_decoder_test.size(0)* output_decoder_test.size(1))
            loss_values_test += loss_test.item()
            
        progress_bar.set_postfix({"Acc_Train": (correct_values_test/total_values_test)*100,
                                    "Loss_Train": (loss_values_test/(batch_test+1))})
