# 谈谈端侧AIPC网络安全的一个场景

> 作者: zartbot  
> 日期: 2024年11月23日 02:24  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247492667&idx=1&sn=4a200edf1903e317045138ae1c939444&chksm=f995f4f9cee27deff5bf9e697bb8dc735223eb8a8795ec9aadc0fbbf5a6e5f19005ae9379b4c#rd

---

昨天一个朋友问到我以前开源的一个zaDNS的项目, 然后想了一些结合LLM**的算法在此分享一下

项目地址: github.com/zartbot/zadns

[《支持AI的ZaDNS服务器》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486069&idx=1&sn=8c49e9e57d26a1225eb1363475f2b610&chksm=f9961ab7cee193a165175997e239fa76fc90c0e5474d87c14326f47c1cc6fda2d4e5791b6e5c&scene=21#wechat_redirect)

当时(2018年)是为了给Cisco做一个完整的AI Infra, 并且在此基础上构建一些AI for network的用例. 然后这个项目拿了CEO大奖并变成一个内部的Startup项目.

      
     
       
         
           
             
                                

                 
                   
已关注
                   **                 
             
             
               关注
           
           
                            **               重播                                         **               分享                                                      **               赞                                     
         
                   
         
                   
         
       
     
     

关闭**

**观看更多**

更多**

**

**

**

*退出全屏*

[**]()

**

   
         
     
       [         视频详情       ]()     
   
 

在几年前NV刚发布Morpheus**的时候我也做过一些关于Nimble的介绍

[《nVidia Morpheus：浅谈AI在网络中的应用》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247485688&idx=1&sn=ccdfaa21419557f0bb16aaed399c505d&chksm=f996183acee1912c14a596247590cc1b4f72ece3ddc401859cf64274002e1ddd9ffc58e68a57&scene=21#wechat_redirect)

有点感慨的是, 当时的sponsor SVP Ravi Chandra也离开思科了, 而当时合作的Fellow JP Vasser也到了NVidia去做AI for Network的head了...

zadns开源是为了处理SDWAN上DNS需要向WAN侧多个链路转发获取最优的结果进行动态路由**的场景以及为了解决某个跨国企业的一个P1故障, 顺手把nimble端侧推理的一些代码埋了进去

[《SDWAN的智能DNS》](http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486086&idx=1&sn=231a54ea9c884da6ff558ae981c61952&chksm=f9961a44cee19352ce1900524c720eb404814c37d7cc6d5a8b65027b8cbb293250dd3886171e&scene=21#wechat_redirect)

![图片](assets/c1fe6cac9ab2.png)

这个项目开始的时间大概是2018年, 当时受限于端侧资源和推理延迟敏的约束, 对于DNS域名安全检测只采用了一个很简单的LSTM的模型进行训练, 把一个安全的问题转换为一个NLP**的问题, 去识别WannaCry这些勒索软件使用的DGA域名. 那个年代的项目大家都还在折腾TensorFlow, Transformer也刚出来没多久.

```
sess = tf.Session()  K.set_session(sess) max_features = 128model=Sequential()model.add(Embedding(max_features, 128,name="inputlayer"))model.add(LSTM(128))model.add(Dropout(0.5))model.add(Dense(128, kernel_initializer='uniform', activation='relu'))model.add(Dense(nb_classes, kernel_initializer='uniform', activation='softmax',name="outputlayer"))model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])model.summary()
```

但是通过这样的方式, 其实就可以构建一些端侧安全的项目, 通过和交换机**路由器联动, 就可以阻塞高危网站的访问.

![图片](assets/5fbd41306d07.png)

然后最近很多年的一些攻击手段直接在DNS报文的Payload里面进行一些隐匿通信绕过各种安全设备, 以及采用各种域名上的字符替换进行诈骗,例如采用1cbc.com/lcbc.com冒充工行.

实际上端侧算力相对足够的情况下, 把LSTM换成多层的transformer是可以尝试一下的,  我们或许可以采用一些< 500M参数的transformer模型来构建这些检测能力, 并且把whois的信息和解析回来的IP Reputation一起做为Embedding 送入模型, 训练方式也非常简单, 找个小的基础模型来做一些FineTune可能就行了, 主要是让模型识别DNS报文的常见格式, 区分一些Txt Record攻击, 然后对于DGA和冒充域名的数据配合大量的正常网站的域名进行训练即可.

大家感兴趣可以去尝试一下吧, 最近工作重心已经到其它计算相关的业务上了, 没有太多精力, 也考虑到业务边界的问题, 就不去卷网络了.