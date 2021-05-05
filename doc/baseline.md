# 百度飞浆智能车挑战赛 - 使用官方Baseline提交你的第一次结果

> [第十六届全国大学生智能车竞赛线上资格赛：车道线检测 - 百度AI Studio - 人工智能学习与实训社区](https://aistudio.baidu.com/aistudio/competition/detail/68)

## 创建项目
1. 在飞桨官网最上方点击 项目 - 创建项目
[image:82D40ABA-9940-4CD0-ACCC-03E381066854-1512-000026D20ED3500E/750862FC-B243-424E-924A-35C7838B9F4E.png]
2. 按如下三图所示创建项目
[image:C4555986-C17B-43B2-BDD0-A4925EC9EF46-1512-000026D6F6C28A36/6671DFAB-9FDD-40B9-B2EE-E5603EA5507C.png]
[image:942F19D1-17E6-4DEE-B5C7-767F69CB14FD-1512-000026E1CE22609F/C498042C-0457-4AC8-ABAF-5CF2B1CD2DFD.png]
[image:7A57780D-410E-4D38-9CD4-0091AF02AA60-1512-000026E7E01855AD/317F72DF-39FF-4D16-BDE9-B0B967E4CBB1.png]
3. 在第三步选择 「添加数据集」，然后搜索“车道线检测Baseline资源包”并添加
[image:B71D9AD1-5AFA-4E6A-9433-427D77DF912F-1512-000026DF2C5CEE9A/78B6ECA8-6E0B-4F04-81DA-02A9FED47185.png]
4. 在我的项目中选择「启动环境」，并选择带有GPU的高级版本
[image:39924069-37C4-4471-B13C-427961AF2618-1512-0000274E79757D18/7B9F2225-C8EF-4AF4-A3B7-001467AFF219.png]
[image:99539B15-1BA7-465F-A242-0CF54415A890-1512-00002754596C4A8F/1BDCB39F-885A-47E0-96C2-DEA3277C94DB.png]

---

## 配置Baseline环境
### 数据准备
如果创建项目时数据集链接正确，则运行环境完全加载完毕后，数据集位于`/home/aistudio/data/data68698/智能车数据集.zip`，可以解压到如下两个文件夹中：
- `data/`：用于本次使用（会在每次进入环境时重置，但可以节省项目启动时间）
-  `work/`：会持久化保存（但是加载环境速度慢）

解压命令如下
```jupyter
!unzip -oq /home/aistudio/data/data68698/智能车数据集.zip -d /home/aistudio/data
```

划分数据集
```jupyter
!python data/make_list.py
```
> 如果报路径相关错误推荐直接使用终端输入指令进行实验

数据文件夹最终结构为
[image:34519639-590E-42D4-A674-2A453DE1FCC6-1512-0000292E749648D4/89B27C67-E612-4A12-926B-542FBE0895E0.png]
- `image` `mask`是用来train的数据集
- `infer`是最终用于predict提交结果的数据集

### 训练
首先clone官方仓库并安装环境

```jupyter
!git clone https://gitee.com/doubleZ0108/PaddleSeg.git
```
> 这个repo有几百M，换成了gitee的仓库，github太慢了

安装到`external-libraries`并引用可以做到持久化安装
```jupyter
!mkdir /home/aistudio/external-libraries
!pip install paddleseg -t /home/aistudio/external-libraries
import sys 
sys.path.append('/home/aistudio/external-libraries')
```

设置环境变量
```jupyter
%set_env CUDA_VISIBLE_DEVICES=0
%set_env PYTHONPATH='/home/aistudio/PaddleSeg'
%cd /home/aistudio
```

修改（重写）yml文件
> 官方给的是使用`configs/deeplabv3p/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml`进行训练，它的base是依靠`configs/_base_/cityscapes.yml`，经过反复的尝试还是会报路径的问题

最终新建了自己的yml文件，注意要更换`train_dataset`中的`dataset_root` `train_path`和`val_dataset`中的`dataset_root` `train_path`
```yml
batch_size: 2
iters: 10000
model:
  type: DeepLabV3P
  backbone:
    type: ResNet50_vd
    output_stride: 8
    multi_grid: [1, 2, 4]
    pretrained: https://bj.bcebos.com/paddleseg/dygraph/resnet50_vd_ssld_v2.tar.gz
  backbone_indices: [0, 3]
  aspp_ratios: [1, 12, 24, 36]
  aspp_out_channels: 256
  align_corners: False
  pretrained: null

train_dataset:
  num_classes: 15
  type: Dataset
  dataset_root: /home/aistudio/work
  train_path: /home/aistudio/work/train_list.txt
  transforms:
    - type: ResizeStepScaling
      min_scale_factor: 0.5
      max_scale_factor: 2.0
      scale_step_size: 0.25
    - type: RandomPaddingCrop
      crop_size: [1024, 512]
    - type: RandomHorizontalFlip
    - type: RandomDistort
      brightness_range: 0.4
      contrast_range: 0.4
      saturation_range: 0.4
    - type: Normalize
  mode: train

val_dataset:
  num_classes: 15
  type: Dataset
  dataset_root: /home/aistudio/work
  val_path: /home/aistudio/work/val_list.txt
  transforms:
    - type: Normalize
  mode: val


optimizer:
  type: sgd
  momentum: 0.9
  weight_decay: 4.0e-5

learning_rate:
  value: 0.00125
  decay:
    type: poly
    power: 0.9
    end_lr: 0.0

loss:
  types:
    - type: CrossEntropyLoss
  coef: [1]
```

执行训练
```jupyter
%cd PaddleSeg/
!python train.py \
       --config configs/mytest/mytest_model.yml \
       --do_eval \
       --use_vdl \
       --save_interval 1000 \
       --save_dir output \
```
其中的`save_interval`为多少轮训练之后保存结果
重新训练时可以增加`--resume_model output/iter_4000`从上次的训练继续执行
[image:A34A1A67-3C58-45E6-8D19-D60F45324761-1512-00002AAD4256BB17/729CE361-2F53-4362-9B54-B016E2E4F7C0.png]

### 预测
```jupyter
%cd PaddleSeg/
!python predict.py \
       --config configs/mytest/mytest_model.yml \
       --model_path output/iter_1000/model.pdparams \
       --image_path ../work/infer \
       --save_dir output/result
```
- `config`：配置的`yml`文件
- `model_path`：训练后的模型
- `image_path`：官方数据集文件夹中的`infer`目录
- `save_dir`：预测结果保存的位置

## 提交结果
将结果打包成`predict.zip`并下载
```jupyter
%cd PaddleSeg/output/result/pseudo_color_prediction
!zip -r -o /home/aistudio/predict.zip ./
```

然后提交到官网（注意压缩包文件名必须为`predict`）
[image:48B2BB1D-7A80-4875-AA44-812DB2CB9302-1512-00002B1353F945CE/1D4D79D5-2601-4018-9BC4-3C7F0C2371F6.png]

到此已经完成了第一次结果的提交🎉

---
## Resources
官方Baseline说明：[官方第十六届全国大学生智能车竞赛线上资格赛-车道线检测Baseline - Baidu AI Studio - 人工智能学习与实训社区](https://aistudio.baidu.com/aistudio/projectdetail/1468678)

