# PKUElectiveCaptcha

elective.pku.edu.cn 补选验证码识别

介个是本渣的第一个机器学习项目噢（小声


### 致谢

该项目基于 [ComradeStukov/TestPKUVerifyCode](https://github.com/ComradeStukov/TestPKUVerifyCode), 借鉴了其中的图片切割算法和特征提取函数

还要感谢各位学长提供的建议和参考代码 orz ...


### 图像处理流程

1. 首先对图片做二值化处理
2. 对图片做两次降噪，去除验证码上的杂点
3. 通过 BFS 找到图片中连续的黑色像素点组成的块，确定各块的宽度范围
4. 对 (3) 的结果进行小处理，确保块的数量为 4，然后进行切割


### 特征提取思路

1. 考虑各像素点在行列间的关系
2. 考虑各像素点与其周围像素点的关系


### 结果与讨论

目前较为适用的算法有 `KNN`, `SVM`, `RandomForest`

单个字母预测准确度：
- 大小写敏感: **98%-98.5%**
- 大小写不敏感: **99.5%**

效率:
- 单个字母预测耗时不超过 **1ms**
- 模型训练时间不超过 **90s**

关于特征提取函数的测试情况见 `./test/feature/\*.csv`


真实条件下，需要一次性准确预测 4 个字母，同时考虑到图片切割的准确度，基于 SVM 算法的模型，大小写敏感的预测准确度为 **95.6%**


### 主要问题

1. 对于包含 `W`, `M` 等较宽字母的验证码，切割效果不佳，因为这些字母容易和周围的字母连在一起，结果被识别为连续块（其实是因为图片分割算法比较渣orz
2. 对于 `J, j`, `K, k` 等大小写形状非常相似的字母，人工识别的准确度不高（眼瞎x


### 文件夹结构

```console
./pku_elective_captcha/
├── download         // 下载到的未经识别的验证码
├── model            // 训练好的模型
├── raw              // 人工识别的验证码
├── segs             // 切割好的验证码字母碎片（数据集）
├── test             // 测试文件夹
├── trash            // 切割效果不好的验证码
├── treated          // 降噪与二值化后的图片
├── classifier.py    // 包含分类器定义、特征提取函数定义、测试函数定义
├── fetch.py         // 抓取验证码
├── preprocess.py    // 图像处理函数的定义与测试
└── util.py          // 通用函数、常量
```


## 证书

[MIT LICENSE](https://github.com/zhongxinghong/PKUElectiveCaptcha/blob/master/LICENSE)
