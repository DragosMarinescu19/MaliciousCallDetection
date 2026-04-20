from models import EnhancedCNN
import torch
model=EnhancedCNN()

model.eval()

x=torch.rand([32,1,128,157])
x.requires_grad=True
pred=model(x)
loss=pred[2].sum()  # Define the loss as depending from only one of the inputs
loss.backward()
for i in range(32):
    if i==2:
        assert (x.grad[i] != 0).any()
    else:
        assert (x.grad[i] == 0).all()
print("Batch leakage successfully prevented")

