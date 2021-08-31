# MatterPort3D

- Heading is defined from the y-axis with the z-axis up (turning right is positive). Camera elevation is measured from the horizon defined by the x-y plane (up is positive). There is also a `newRandomEpisode` function which only requires a list of scanIds, and randomly determines a viewpoint and heading (with zero camera elevation). 
- For agent `n`, navigable locations are given by `getState()[n].navigableLocations`
- Index 0 always contains the current viewpoint (i.e., the agent always has the option to stay in the same place). As the navigation graph is irregular, the remaining viewpoints are sorted by their angular distance from the centre of the image, so index 1 (if available) will approximate moving directly forward. For example, to turn 30 degrees left without moving (keeping camera elevation unchanged): 

```python
sim.makeAction([0], [-0.523599], [0])
```

- At any time the simulator state can be returned by calling `getState`. The returned state contains a list of objects (one for each agent in the batch), with attributes as in the following example:

```python
[
  {
    "scanId" : "2t7WUuJeko7"  // Which building the agent is in
    "step" : 5,               // Number of frames since the last newEpisode() call
    "rgb" : <image>,          // 8 bit image (in BGR channel order), access with np.array(rgb, copy=False)
    "depth" : <image>,        // 16 bit single-channel image containing the pixel's distance in the z-direction from the camera center 
                              // (not the euclidean distance from the camera center), 0.25 mm per value (divide by 4000 to get meters). 
                              // A zero value denotes 'no reading'. Access with np.array(depth, copy=False)
    "location" : {            // The agent's current 3D location
        "viewpointId" : "1e6b606b44df4a6086c0f97e826d4d15",  // Viewpoint identifier
        "ix" : 5,                                            // Viewpoint index, used by simulator
        "x" : 3.59775996208,                                 // 3D position in world coordinates
        "y" : -0.837355971336,
        "z" : 1.68884003162,
        "rel_heading" : 0,                                   // Robot relative coords to this location
        "rel_elevation" : 0,
        "rel_distance" : 0
    }
    "heading" : 3.141592,     // Agent's current camera heading in radians
    "elevation" : 0,          // Agent's current camera elevation in radians
    "viewIndex" : 0,          // Index of the agent's current viewing angle [0-35] (only valid with discretized viewing angles)
                              // [0-11] is looking down, [12-23] is looking at horizon, is [24-35] looking up
    "navigableLocations": [   // List of viewpoints you can move to. Index 0 is always the current viewpoint, i.e. don't move.
        {                     // The remaining valid viewpoints are sorted by their angular distance from the image centre.
            "viewpointId" : "1e6b606b44df4a6086c0f97e826d4d15",  // Viewpoint identifier
            "ix" : 5,                                            // Viewpoint index, used by simulator
            "x" : 3.59775996208,                                 // 3D position in world coordinates
            "y" : -0.837355971336,
            "z" : 1.68884003162,
            "rel_heading" : 0,                                   // Robot relative coords to this location
            "rel_elevation" : 0,
            "rel_distance" : 0
        },
        {
            "viewpointId" : "1e3a672fa1d24d668866455162e5b58a",  // Viewpoint identifier
            "ix" : 14,                                           // Viewpoint index, used by simulator
            "x" : 4.03619003296,                                 // 3D position in world coordinates
            "y" : 1.11550998688,
            "z" : 1.65892004967,
            "rel_heading" : 0.220844170027,                      // Robot relative coords to this location
            "rel_elevation" : -0.0149478448723,
            "rel_distance" : 2.00169944763
        },
        {...}
    ]
  }
]
```



# R2R code

[toc]



-----

## agent.py

### class BaseAgent

基础的agent类，其中 rollout 函数需要被其他类型的agent重写。

1. results：{'insrt_id', 'trajectory'}
2. def rollout: 返回字典列表：[{'instr_id', 'path:[(viewpointId, heading_rad, elevation_rad)]'}]

### class StopAgent

停止的agent，重写 rollout 函数，返回 traj

### class RandomAgent

机器随机选择一个方向，然后向前走５步，然后停止。随机x

### class ShortestAgent

总是跟着最短的路径走。

### class Seq2SeqAgent

基于带有attention模块的LSTM的seq2seq的机器。

```python
# For now, the agent can't pick which forward move to make - just the one in the middle
model_actions = ['left', 'right', 'up', 'down', 'forward', '<end>', '<start>', '<ignore>']
env_actions = [
  (0,-1, 0), # left
  (0, 1, 0), # right
  (0, 0, 1), # up
  (0, 0,-1), # down
  (1, 0, 0), # forward
  (0, 0, 0), # <end>
  (0, 0, 0), # <start>
  (0, 0, 0)  # <ignore>
]
feedback_options = ['teacher', 'argmax', 'sample']
```



#### def _sort_batch(self, obs)

从一系列的observatio中提取特征，并且根据长度来排序（降序排序）

mask 用于将短序填充为最长（？）

方便pytorch打包成一个Variable

#### def _feature_variable(self, obs)

将预先处理好的特征提取到Variable里。

#### def _teacher_action(self, obs, ended)

提取teacher actions到Variable里。

#### def rollout(self)

!!! 还没看懂，涉及强化学习的知识了！需要恶补。

```python
# Save trajectory output
for i,ob in enumerate(perm_obs):
    if not ended[i]:
        traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
```

最后保存trajectory output，并返回。

#### def test()

#### def train()

#### def save()

将encoder和decoder的权重保存下来。

#### def load()

读取encoder和decoder的权重。

-----

## env.py

### class EnvBatch

对于matterport3D环境的简单包装，使用离散的viewpoint和预训练特征。

tsv文件的title包括：scanId, viewpointId, image_w, image_h, vfov, features。（默认image_w=640, image_h= 480, vfov=60)。

self.sim = MatterSim.Simulator()

####  def getStatus(self)

获得预计算好的image features。

return: feature_states

#### def makeActions(self, action)

每个action应该是一个(index, heading, elevation)的三元组。

在sim中做出行动。

#### def makeSimpleActions(self, simple_indeices)

做出简单的行为，包括： 0-forward, 1-turn left, 2-turn right, 3-look up, 4-look down。所有的viewpoint的改变都是30度。注意，forward, lookup 和 lookdown可能不会成功，因为这可能会超出环境的限制。——注意check state。

self.makeAction(actions)



### class R2RBatch

使用离散的viewpoint和pretrained featured的，应用于R2R任务。

self.data = [{'instr_id', 'instructions','instr_encoding'}] 

注意，这里'instr_id' = '%s_%d' % (item['path_id'], j)

```python
if tokenizer:
	new_item['instr_encoding'] = tokenizer.encode_sentence(instr)
```

#### def _load_nav_graph(self)

为每个scan加载连接图，对于推理最短路径有用.

`self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))`

`self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))`

#### def _next_minibatch(self)

获得下一个minibatch，更新self.ix

#### def　_shortest_path_action(self, state, goalViewpointId)

决定下一个最短到达目的地的行为，用于监督训练。

首先判断是否能够看到下一个viewpoint：在移动前先面向那个viewpoint（向左转or向右转）。

如果看不到下一个viewpoint，则先矫正当前camera的位姿，或者再自行决定是向左转or向右转。

#### def _get_obs(self)

(obs = observation)

```python
def _get_obs(self):
    obs = []
    for i,(feature,state) in enumerate(self.env.getStates()):
        item = self.batch[i]
        obs.append({
            'instr_id' : item['instr_id'],
            'scan' : state.scanId,
            'viewpoint' : state.location.viewpointId,
            'viewIndex' : state.viewIndex,
            'heading' : state.heading,
            'elevation' : state.elevation,
            'feature' : feature,
            'step' : state.step,
            'navigableLocations' : state.navigableLocations,
            'instructions' : item['instructions'],
            'teacher' : self._shortest_path_action(state, item['path'][-1]),
        })
        if 'instr_encoding' in item:
            obs[-1]['instr_encoding'] = item['instr_encoding']
    return obs
```

#### def reset(self)

加载一个新的minibatch / episodes。

获得新的scanIds, viewpointIds, headings，然后去更新`self.env.newEpisodes()`。

return: self._get_obs()

#### def step()

做出决定,使用 `self.env.makeActions(actions)`

return: self._get_obs()

-----

## eval.py

用于评价agent trajectories。

### class Evaluation

提交的格式应该满足：

`[{'instr_id': string, 'trajectory':[(viewpoint_id, heading_rads, elevation_rads),] } ]`

#### def _get_nearest(self, scan, goal_id, path)

获取最近距离的id

#### def _score_item(self, instr_id, path)

基于trajectories最后的重点位置来计算error，并且计算最近的位置（*oracle stopping rule?*）

```python
self.score[{
'nav_errors', # 最后停下来的点是否在终点的3m以内
'oracle_errors', # 最后停下来的最近一个点是否在终点的3m以内
'trajectory_lengths', # 总路径长度
'shortest_path_lengths' # 真值的最短路径长度
}]
```

#### def score(self, output_file)

读取文件，得到所有路径的score

```python
score_summary ={
    'length': np.average(self.scores['trajectory_lengths']),
    'nav_error': np.average(self.scores['nav_errors']),
    'oracle success_rate': float(oracle_successes)/float(len(self.scores['oracle_errors'])),
    'success_rate': float(num_successes)/float(len(self.scores['nav_errors'])),
    'spl': np.average(spls)
}
```

其中，spl的计算公式如下：

```python
spls = []
for err,length,sp in zip(self.scores['nav_errors'],self.scores['trajectory_lengths'],self.scores['shortest_path_lengths']):
    if err < self.error_margin:
        spls.append(sp/max(length,sp))
    else:
        spls.append(0)
```

### def eval_simple_agents()

计算三种简单的baseline的结果：stop \ shortest \ random

### def eval_seq2seq()

计算两种seq2seq模型：teacher-based 和 student-based 在两个数据集上的结果： val_seen 和 val_unseen.

## model.py

### class EncoderLSTM

编码环境指令，返回隐藏状态信息（为了注意力方法）以及一个解码器的初始状态。

### class SoftDotAttention

*Soft Dot Attention*

### class AttnDecoderLSTM

