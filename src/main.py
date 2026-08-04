import torch


class adam:
   def __init__(self, params, lr, beta1, beta2,eps):
      self.params = list(params)
      self.lr = lr
      self.beta1 = beta1
      self.beta2 = beta2
      self.eps = eps

      # Adam has independent state for every parameter tensor.
      self.first_moments = [
            torch.zeros_like(param) for param in self.params
      ]
      self.second_moments = [
            torch.zeros_like(param) for param in self.params
      ]

      self.steps = [0 for _ in self.params]

   @torch.no_grad()
   def step(self):
      """
      Iterate along all the paramters and update them. 
      For each parameter I should maintain in memory the momentum m1 and the magnitude v1.
      """
      for i, param in enumerate(self.params):
         grad = param.grad
         self.steps[i] += 1
         t = self.steps[i]

        
         m0 = self.first_moments[i]
         v0 = self.second_moments[i]

         m1 = self.beta1 * m0 + (1 - self.beta1) * grad
         v1 = self.beta2 * v0 + (1 - self.beta2) * grad**2

         self.first_moments[i] = m1
         self.second_moments[i] = v1

         m_hat = m1 / (1.0 - self.beta1**t)
         v_hat = v1 / (1.0 - self.beta2**t)

         update = self.lr * (m_hat / (torch.sqrt(v_hat) + self.eps))

         param -= update



   def zero_grad(self):
      for param in self.params:
         param.grad = None







"""
Conceptually identical to adam, but at the end in addiction to the update we also contract the parameter
"""
class adamw:
   def __init__(self, params, lr, beta1, beta2, lamb,eps):
      self.params = list(params)
      self.lr = lr
      self.beta1 = beta1
      self.beta2 = beta2
      self.eps = eps
      self.lamb = lamb
      # Adam has independent state for every parameter tensor.
      self.first_moments = [
            torch.zeros_like(param) for param in self.params
      ]
      self.second_moments = [
            torch.zeros_like(param) for param in self.params
      ]

      self.steps = [0 for _ in self.params]

   @torch.no_grad()
   def step(self):
      """
      Iterate along all the paramters and update them. 
      For each parameter I should maintain in memory the momentum m1 and the magnitude v1.
      """
      for i, param in enumerate(self.params):
         grad = param.grad
         self.steps[i] += 1
         t = self.steps[i]

        
         m0 = self.first_moments[i]
         v0 = self.second_moments[i]

         m1 = self.beta1 * m0 + (1 - self.beta1) * grad
         v1 = self.beta2 * v0 + (1 - self.beta2) * grad**2

         self.first_moments[i] = m1
         self.second_moments[i] = v1

         m_hat = m1 / (1.0 - self.beta1**t)
         v_hat = v1 / (1.0 - self.beta2**t)

         update = self.lr * (m_hat / (torch.sqrt(v_hat) + self.eps) + self.lamb * param)

         param -= update



   def zero_grad(self):
      for param in self.params:
         param.grad = None


