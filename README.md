# Federated Optimizers in PyTorch
## What is Federated Learning?
> Federated Learning is a distributed machine learning approach which enables model training on a large corpus of decentralized data. 

_From Towards Federated Learning at Scale: System Design_


## Federated Averaging 
### Description
- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/pdf/1602.05629.pdf)
<img width="364" alt="Screenshot 2019-07-12 at 14 14 07" src="https://user-images.githubusercontent.com/26603883/61127621-13820c00-a4b0-11e9-80aa-9900cca24956.png">

### Usage
```
optimizer = optimizers.FederatedAvgServer(model.parameters())
optimizer.zero_grad()
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

## Federated Stochastic Variance Reduced Gradient Decent (FSVRG)
### Description
- [Federated Optimization: Distributed Machine Learning for On-Device Intelligence](https://www.maths.ed.ac.uk/~prichtar/papers/federated_optimization.pdf)
<img width="764" alt="Screenshot 2019-07-18 at 20 07 42" src="https://user-images.githubusercontent.com/26603883/61481211-ee9e0500-a997-11e9-8dce-aba143c77691.png">
<img width="751" alt="Screenshot 2019-07-18 at 20 07 33" src="https://user-images.githubusercontent.com/26603883/61481269-2442ee00-a998-11e9-897a-d8324860e6f5.png">

### Usage

#### I am currently working on this and I hope I will be able to have at least a rough outline online with in the next 2 days.
