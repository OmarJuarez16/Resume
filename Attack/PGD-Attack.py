def pgd_linf(model, X, y, epsilon=8/255, alpha=2/255, mode="Train" randomize=False):
    
    if mode == "Test": 
      alpha = epsilon * 2.5 / 100
      num_iter = 100
    else: 
      num_iter = 7
      
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
        
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X.float() + delta.float()), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()
