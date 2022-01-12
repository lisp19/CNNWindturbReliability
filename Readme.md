# 文件夹结构说明

工作空间根目录：

+ 代码文件

+ Source文件夹：用于存放训练标签和原始csv数据

  存放格式：Source/train_labels.csv, Source/011/xxxx.csv

  即训练标签存放在Source文件夹根路径，每一个风机的csv数据存放在以风机编号命名的Source文件夹子路径中，子路径长度保持3位，前补0，如012，003

+ Data文件夹：用于存放处理后的数据文件

+ Model文件夹：用于存放基于模型的迁移学习中源领域模型

+ Model_transfer文件夹：用于存放迁移学习训练结果

+ Prediction文件夹：用于存放预测结果

***复现代码需保持文件夹结构不变，或更改相对路径***

# 项目运行环境说明

+ 操作系统: Windows 11 X86_64/ Pop!_OS 21.10（均可）
+ Python 版本: 3.7.5 64bit (Anaconda)
+ Tensorflow 及tf.keras 版本: 2.5.0
+ CUDA 版本: 11.1
+ CUDNN 版本: 8.1.01
+ 依顺序完成上述环境搭建后，其余依赖保持默认

# 代码文件说明

以下代码为预处理/中间文件生成代码，项目期间完成后运行一次多次使用。如果以中间文件为基础，部分代码文件无需运行。（后文有详细说明）

+ Data_Loading.py：读取csv文件并整理为字典, 整理、归一化后输出训练集和测试文件："Data/Train.npy", "Data/Test.npy","Data/Test0.pkl","Data/Test1.pkl"
+ Data_Predicting.py: 读取待预测csv整理为字典，保存为Data/datasets_predicting.pkl
+ Data_Reload_Predicting.py: 以Data/datasets_predicting.pkl 为基础，整理归一化并生成预测用文件，保存为Data/Predicting.pkl

以下代码为模型训练代码：

+ Model.py: 迁移学习源领域模型训练代码，前置文件Data/Train.npy, Data/Test.npy, Data/Test0.pkl, Data/Test1.pkl，生成模型"Best_Model.hdf5"保存至Model文件夹。***实际操作中会在每个epoch后保存最优验证模型，因此无需运行150个epoch，约30epoch即可收敛***。
+ Model_transfer.py: 迁移学习目标领域训练代码，前置文件Model/Best_Model.hdf5,Data/Test.npy, Data/Test0.pkl, Data/Test1.pkl, 生成模型“Best_Model_transfer.hdf5”保存至Model_transfer文件夹

以下模型为测试/预测代码：

+ Test.py: 对于给定模型，在12号风机给出的30%数据上进行测试，输出准确率。前置文件Model_transfer/Best_Model_transfer.hdf5, Data/Test0.pkl, Data/Test1.pkl, 无文件输出，测试结果在标准输出中显示
+ Predicting.py: 对于给定模型，对12号风机剩余70%的数据给出预测标签，前置文件Model_transfer/Best_Model_transfer.hdf5, Data/Predicting.pkl, 输出预测文件csv Prediction/Result.csv
