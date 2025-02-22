"""
    def predict(self, state):
        pos = state.gamestate_to_statedict()["pacman_coord"]
        legal_action = get_valid_moves_pacman(pos, state)

        log_act_probs, value = self.ValueNet(state)
        act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_action, act_probs[legal_action])

        value_float = value.float()

        # act_probs = [] (length = 5)
        # zip ( * , * )

        return act_probs, value_float

    def predict(self, state):
        pos = state.gamestate_to_statedict()["ghosts_coord"]
        legal_actions = get_valid_moves_pacman(pos, state)

        log_act_probs, value = self.ValueNet(state)

        act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs_new = {}
        for action, index in enum(legal_actions):
            act_probs_new[action]=[]
            act_probs_new[action].append(act_probs[index*3])
            act_probs_new[action].append(act_probs[index*3+1])
            act_probs_new[action].append(act_probs[index*3+2])
        act_probs_new = zip(legal_actions, act_probs_new[legal_actions])

        # act_probs_new = {action=[ , , ]: [ , , ]}
        # zip ([ , , ], [ , , ])

        value_float = value.float()

        return act_probs_new, value_float

    def _train_step(self, state_batch, mcts_probs, winner_batch, lr):
        state_batch = Variable(torch.FloatTensor(state_batch))
        mcts_probs = Variable(torch.FloatTensor(mcts_probs))
        winner_batch = Variable(torch.FloatTensor(winner_batch))

        self.optimizer_pacman.zero_grad()

        log_act_probs, value = self.policy_value_net_pacman(state_batch)
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss

        loss.backward()
        self.optimizer_pacman.step()

        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        
        return loss.item(), entropy.item()
"""

# 回合制MCTS, 搜索到回合结束, 开始倾向于广度搜索, 后面倾向于深度搜索
# 通过深度残差网络给出q和pi
# 探索次数说明有价值

# 网络结构, 公共头, 策略头, 价值头
# 网络就是参数 \theta

# agent = Agent()
# for iteration:
#   trajs=[]
#   for episode: 获取经验
#       traj.append(play())
#   learn() 更新参数theta
#   play()
#
# def play:
#   traj = []
#   while True:
#       action, prob_distri = agent.decide
#       step(action)
#       if done: break
#       (p, state, bonus, who)
#
# def decide:
#   agent.search() # 多次MCTS
#   p = count[action]/sum(count[action])
#   a = random_choice(p)
#   return (a, p)
#
# def search:
#   # 递归函数
#   if done: return v
#   if policy is empty:
#       action_prob, value = net(state)
#       self.policy = softmmax(action_prob.correct())
#       return Vs
#       # 此后策略不会变化，但是价值会更新
#   else:
#       # PUCT
#       # find max(\lambda \pi + self.q) # 初始q置零
#       v = search(max)
#       self.count + 1
#       self.q += (v-q)/n # 增量更新
#       return v
#
# def learn:
#   for batches:
#       net_fit(traj) # 训练theta
#   reset() # 只保留theta