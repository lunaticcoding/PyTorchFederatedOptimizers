# Federated Optimizers in PyTorch
## Federated Averaging 
### Description
- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/pdf/1602.05629.pdf)
<img width="364" alt="Screenshot 2019-07-12 at 14 14 07" src="https://user-images.githubusercontent.com/26603883/61127621-13820c00-a4b0-11e9-80aa-9900cca24956.png">

### Usage
```
optimizers = optimizers.FederatedAvgServer(model.parameters())
optimizers.zero_grad()
loss_fn(model(input), target).backward()

# On every client then do
optimizer_client = optimizers.FederatedAvgClient(model.parameters(), lr=0.1)
optimizer_client.zero_grad()
loss_fn(model(input), target).backward()
optimizer_client.step()
nk_grad = (n_training_examples, model.parameters())

# Send nk_grad from clients (1 to l) to the server
list_nk_grad = [nk_grad1, ..., nk_gradl]
optimizers.step(list_nk_grad)
# Redistribute updated model.parameters() from server to clients
```
