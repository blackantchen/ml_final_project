# 机器学习纳米学位毕业项目

## 猫狗大战 (Dogs vs. Cats)

## Source code list

- preprocess_data.ipynb: 数据预处理和特征提取，3个模型特征提取共耗时约10分钟（aws p3.2xlarge 实例）
- find_outlier.ipynb: 从训练集刷选异常值
- train_model.ipynb: 模型训练，对测试机预测并生成csv
- model_history_show.ipynb: 从history文件中读取，并展示各模型的训练曲线
- data_explorer.ipynb: 训练集数据探索，结果可视化

## Report

- **MLND_Machine_Learning_Capstone_Project_Report.pdf**: 毕业项目报告

## The main libraries used by the project

- numpy
- pandas
- matplotlib
- keras
- tensorflow
- sklearn
- tqdm
- opencv-python
- h5py

## The running time of all code

About 30 minutes, on AWS p3.2xlarge

## Notice

Before running the notebook, download the dataset from kaggle first: [Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)