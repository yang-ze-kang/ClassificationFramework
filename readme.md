# 图像分类Pytorch代码框架

在configs中新建config文件选择模型、数据、优化器参数\
在models中构建模型\
train.py/test.py已封装好好无需改动\

## 代码框架update日志
- 2022-5-11 初次提交\
model:vit\
优化器默认Adam，学习率衰减默认余弦\
数据加载方式默认从txt文件读取