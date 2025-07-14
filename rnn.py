import torch
import torch.nn as nn

hidden_size = 2
iterations = 100
learning_rate = 0.1

#Input 
x = torch.tensor([1.0, 2.0, 3.0]) 
#Convert to PyTorch(sequence_length, batch_size, input_size) format
x = x.view(3, 1, 1) 

#Observed output (for training)
y = torch.tensor([2.0, 3.0, 4.0])
#Convert to PyTorch (sequence_length, batch_size, input_size) format
y = y.view(3, 1, 1) 


 #creates RNN
rnn   = nn.RNN(input_size=1, hidden_size=hidden_size) 
#linear layer to connect hidden layer's to final output
decode = nn.Linear(hidden_size, 1)

#Stochastic Gradient Descent is the optimizer to train model 
opt     = torch.optim.SGD(list(rnn.parameters()) + list(decode.parameters()), lr=learning_rate)
#Mean Squared Error is the loss function to calculate the loss between predicted and observed
criterion = nn.MSELoss()

# 4) Training loop
for step in range(iterations):
    #reset the gradients in between training batches
    opt.zero_grad() 
    #resets the RNN hidden layer states (vector of hidden outputs) in between training batches
    h0    = torch.zeros(1, 1, hidden_size)
    #gets output of hidden layer
    out, _ = rnn(x, h0)
    #converts the output state into a prediction 
    pred   = decode(out)            
    #calculates loss based on actual observed output (Mean Squared Error)
    loss   = criterion(pred, y)     
    #Computes the gradients based on the loss
    loss.backward()
    #updates the models weights to minimize loss
    opt.step()
    if step % 20 == 0:
        print(f"Step {step:3d}  Loss {loss.item():.4f}")

# 5) Check final prediction
print("Predicted sequence:", pred.squeeze().detach().numpy())