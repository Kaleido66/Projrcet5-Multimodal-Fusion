# 实验五-多模态情感分析

给定配对的文本和图像，预测对应的情感标签。三分类任务：positive, neutral, negative。

## Setup

安装本实验需要的包，在终端运行以下命令：

```python
pip install -r requirements.txt
```

## Repository Structure

本实验的文件结构如下：

```python
|-- data	# 数据集
|-- bert-base-chinese  # 本地bert路径
	|-- config.json
	|-- merges.txt
	|-- tokenizer.json
	|-- tokenizer_config.json
	|-- vocab.json
	|-- tf_model.h5
|-- main.py	# 完整的项目代码
|-- train.txt	# 训练集
|-- test_without_label.txt	# 测试集
|-- test.txt	# 预测结果
|-- README.md
|-- requirements.txt
```

## Usage

在终端输入

```python
python main.py
```

**注意：**

代码中的文件路径均为本地路径，若需要运行则需要改成终端上对应的路径，如数据路径、`./bert-base-chinese`文件路径。

本次实验的运行时间比较长，最长的需要将近一天的时间才能得出结果。

**在bert-base-chinese文件夹中，运行时需先将tf_model.h5放置在文件夹内才可运行，否则会出现报错。**