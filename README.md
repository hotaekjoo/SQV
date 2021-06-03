# SQV-pytorch
The official implementation of the paper `A Swapping Target Q-Value Technique for Data Augmentation in Offline Reinforcement Learning`  
Our implementation is implemented under [fujimoto's BCQ implementation](https://github.com/sfujim/BCQ), and the additional is abstracted in `Main Implementation` section.


## Main Implementation
### Swapping Q-Value ([Code](https://github.com/nips-anonymous-1955/SQV/blob/main/discrete_SQV/discrete_SQV.py#L120))
```python
def train_aug_sw(self, replay_buffer):
    # Sample replay buffer
    state, action, next_state, reward, done, state_aug, next_state_aug = replay_buffer.sample(aug_type="aug")

    # Compute the target Q value
    with torch.no_grad():
        q, imt, i = self.Q(next_state)
        q_aug, imt_aug, i_aug = self.Q(next_state_aug)

        imt = imt.exp()
        imt = (imt / imt.max(1, keepdim=True)[0] > self.threshold).float()

        imt_aug = imt_aug.exp()
        imt_aug = (imt_aug / imt_aug.max(1, keepdim=True)[0] > self.threshold).float()

        # Use large negative number to mask actions from argmax
        next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)
        next_action_aug = (imt_aug * q_aug + (1 - imt_aug) * -1e8).argmax(1, keepdim=True)

        q, imt, i = self.Q_target(next_state)  # q, imt, i = torch.Size([32, 6])
        q_aug, imt_aug, i_aug = self.Q_target(next_state_aug)

        target_Q = reward + done * self.discount * q.gather(1, next_action).reshape(-1, 1)
        target_Q_aug = reward + done * self.discount * q_aug.gather(1, next_action_aug).reshape(-1, 1)

    # Get current Q estimate
    current_Q, imt, i = self.Q(state)
    current_Q_aug, imt_aug, i_aug = self.Q(state_aug)

    current_Q = current_Q.gather(1, action)
    current_Q_aug = current_Q_aug.gather(1, action)

    # Compute Swapped Q loss
    q_loss = F.smooth_l1_loss(current_Q, target_Q_aug) + F.smooth_l1_loss(current_Q_aug, target_Q)
    i_loss = F.nll_loss(imt, action.reshape(-1)) + F.nll_loss(imt_aug, action.reshape(-1))
    Q_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean() + 1e-2 * i_aug.pow(2).mean()

    # Optimize the Q
    self.Q_optimizer.zero_grad()
    Q_loss.backward()
    self.Q_optimizer.step()
```


### Image Augmentation ([Code](https://github.com/nips-anonymous-1955/SQV/blob/main/discrete_SQV/utils.py#L109))
```python
def aug_trans(self, imgs, out=84):
    imgs = nn.ReplicationPad2d(self.image_pad)(imgs)
    n, c, h, w = imgs.shape
    crop_max = h - out + 1
    w1 = torch.randint(0, crop_max, (n,))
    h1 = torch.randint(0, crop_max, (n,))
    cropped = torch.empty((n, c, out, out), dtype=imgs.dtype)

    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
    return cropped, abs(8 - (w1 + h1))
```

## Reproduction
### Requirements
- Python 3.6
- pytorch 1.4
- gym

### Run Code
To begin a behavioral policy (DQN) needs to be trained by running:
```bash
python main.py --train_behavioral --env AsterixNoFrameskip-v0
```
This will save the PyTorch model. A new buffer can then be collected by running:
```bash
python main.py --generate_buffer --env AsterixNoFrameskip-v0
```
Finally train vanilla BCQ or our SQV by running:
```bash
python main.py --env AsterixNoFrameskip-v0
``` 
```
python main.py --env AsterixNoFrameskip-v0 --sqv 
```

## Disclaimer
Our implemtation is under [fujimoto's BCQ baselines](https://github.com/sfujim/BCQ), and we follow the MIT License.
```
MIT License

Copyright (c) 2020 Scott Fujimoto

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
