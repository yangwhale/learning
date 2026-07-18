# 支持AI的ZaDNS服务器

> 作者: zartbot  
> 日期: 2021年7月23日 01:56  
> 原文: http://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247486069&idx=1&sn=8c49e9e57d26a1225eb1363475f2b610&chksm=f9961ab7cee193a165175997e239fa76fc90c0e5474d87c14326f47c1cc6fda2d4e5791b6e5c#rd

---

最近处理某个客户的`P1`故障，顺手写了一个带`域名路由`的`DNS服务器`，然后夹带了一些基于`AI网络安全`功能，框架集成了Tensorflow的支持，并给您了一个详细的从`模型训练`到`部署上线`的全流程，请参考本文稍后的章节，基本框架的代码已开源，欢迎大家一起来构建一个更加安全快速的网络。

https://github.com/zartbot/zadns

### ZaDNS支持功能

#### ZaDNS部署环境

通常我们会遇到如下一些情况，由于不同的运营商或者应用对于DNS的优化解析,使得流量在请求不同的DNS服务器时会有不同的结果,同时还存在一些DNS有`侵犯隐私`和`污染干扰`的情况,因此很多企业都会使用`基于域名`的DNS路由功能，如下图所示

![图片](assets/a096fe1563d2.png)

但是主机基本上只能设置一主一备两个DNS服务器，在多平台上部署Bind也是一个麻烦事，因此写了一个zaDNS的开源项目来实现这些基本功能，您可以将其部署在路由器上，也可以直接在主机上部署，整个项目基于Golang构建，支持Windows、Linux、MAC主机侧部署

在Linux版本中还内置了支持DGA识别的一个神经网络，利用Tensorflow进行推理，Tensorflow在MAC和Windows上留给大家自己研究吧，毕竟我自己部署放在一个基于Linux的路由器上就好了.

#### 域名路由

在`config/route.cfg`中，您可以定义DNS路由规则

```
cisco.com:  8.8.8.8,4.4.4.4google.com: 8.8.8.8taobao.com: 223.5.5.5,223.6.6.6
```

默认的DNS服务器在`config/server.cfg`中定义.

路由的功能很简单，将DNS Domain字符串倒序，然后放入一个Radix-Tree中，再针对Query的字符串进行LPM

```
func ReverseString(req string) string { if req[len(req)-1:] != "." {  req = req + "." } rns := []rune(req) for i, j := 0, len(rns)-1; i < j; i, j = i+1, j-1 {  rns[i], rns[j] = rns[j], rns[i] } return string(rns)}func (p *Proxy) updateRadixTree() { for k, v := range p.route {  p.routeTree.Insert(ReverseString(k), v) }}func (p *Proxy) DomainRouteLookup(req string) []string { result := make([]string, 0) _, v, found := p.routeTree.LongestPrefix(ReverseString(req)) if found {  return v.([]string) } return result}
```

您也可以利用这个Radix-Tree来将Alexa的前一百万个知名域名作为白名单放入，通过查询来放行DNS流量

返回的是一个string列表，包含了该域名对应的DNS服务器列表，我们可以采用随机查询的方法，或者顺序查询查到退出

```
//RandomLookup is based on shuffled serverlist sequencefunc (p *Proxy) RandomLookup(msg *dns.Msg, serverList []string) (*dns.Msg, error) { tServerList := make([]string, 0) tServerList = append(tServerList, serverList...) rand.Shuffle(len(tServerList), func(i, j int) { tServerList[i], tServerList[j] = tServerList[j], tServerList[i] }) for _, s := range tServerList {  resp, err := Lookup(msg, s)  if err == nil {   return resp, nil  } } return nil, fmt.Errorf("serverNotAvailable")}//Lookup is used to get the response from external serverfunc Lookup(msg *dns.Msg, server string) (*dns.Msg, error) { c := new(dns.Client) c.Net = "udp" resp, _, err := c.Exchange(msg, server) if err != nil {  return nil, err } return resp, nil}
```

#### 广告过滤

有些时候我们家里的电视机带广告太烦人，也需要过滤一些域名，例如已经有好心人收集的如下列表

```
https://raw.githubusercontent.com/vokins/yhosts/master/hosts
```

这个时候您可以配置`config/hosts.cfg`，将这些解析结果纳入，ZaDNS会将其放入本地Cache中，直接响应客户端的请求，完成对特定域名的过滤。

#### 地理位置和BGP AS号关联

通过集成`GeoIP2`数据库，ZaDNS会对返回的`A/AAAA`记录进行地理位置信息查询，这样我们就可以对地理位置信息进行比对，筛除较远的或者风险相对较高的地区，实现对主机的初级安全保护。

同时GeoIP2还支持`BGP-AS`关联，那么顺便查询的时候把ASN也拿出来，这样就可以识别CDN的地址了，例如当返回多个A记录时，我们可以分析从不同的运营商DNS返回的A记录进行拼合和测试，或者根据AS号, 影响内网的主机选择合适的服务器实现多广域网路径的流量工程.

当然GeoIP数据库里面有一个大问题，把港澳台说成是一个国家，这种问题得治一下

```
//Lookup is used to find IP location in GeoIPDBfunc (g *GeoIPDB) Lookup(ipAddr string) GeoLocation { var r GeoLocation ip := net.ParseIP(ipAddr) if ip == nil {  return r } c, _ := g.CityDB.City(ip) asn, _ := g.ASNDB.ASN(ip) if c.City.GeoNameID != 0 {  r.City = c.City.Names["en"] } if len(c.Subdivisions) > 0 {  if c.Subdivisions[0].GeoNameID != 0 {   r.Region = c.Subdivisions[0].Names["en"]  } } if c.Country.GeoNameID != 0 {  r.Country = c.Country.Names["en"] } if r.Country == "Hong Kong" {  r.Country = "China"  r.Region = "Hong Kong"  r.City = "Hong Kong" } if r.Country == "Macau" || r.Country == "Macao" {  r.Country = "China"  r.Region = "Macau"  r.City = "Macau" } if r.Country == "Taiwan" {  r.Country = "China"  r.Region = "Taiwan" } r.Latitude = c.Location.Latitude r.Longitude = c.Location.Longitude r.ASN = asn.AutonomousSystemNumber r.SPName = asn.AutonomousSystemOrganization return r}
```

### Tensorflow + DNS

训练用数据集和Tensorflow训练的代码在`utils/dga`中

#### DGA的来历

现在网络安全的最大威胁来自于勒索，常见的做法是让受控主机和`C&C`（Command and Control，简称C2）主机进行通信并接受控制，勒索的触发都需要涉及如何和C2通信。C2自然会是那种过街老鼠，东躲西藏，通常的做法是按照某种特定的算法生成一个DNS域名，通过查询获得地址，例如`wannacry`所采用的域名:

![图片](assets/26711edcab74.png)

使用RNN可以非常容易的区分这些网站，这也是很多做安全的DNS服务器都支持的功能，例如Cisco的Umbrella，但是这些安全的服务器并没有很好的基于用户地理位置，广域网情况实现很好的路径优选，那么将这些安全功能前置到主机或接入路由器上便是一个更好的选择，这也是ZaDNS使用的一个主要场景

#### 数据获取

白名单数据采用`Alexa top 1m`的数据集,可以在如下连接访问到:

http://s3.amazonaws.com/alexa-static/top-1m.csv.zip

DGA等算法产生的黑名单数据可以通过360获得:

https://data.netlab.360.com/feeds/dga/dga.txt

数据的预处理很简单，我们将ASC String转换为整数

```
alexa = pd.read_csv("./top-1m.csv", names=["rank","name"])alexa['label'] = "normal"alexa['b_label'] = 0dga360 = pd.read_csv("./dga.txt",delim_whitespace=True,header=16,names=["label","name","a","b","c","d"])dga360['b_label'] = 1a = alexa[["name","label","b_label"]]b = dga360[["name","label","b_label"]]dga = pd.concat([a,b])#change label name to int valuelabel_set = set(dga['label'])label_map = {}i = 0for item in set(label_set):    label_map[item] = i    i=i+1    dga['label_v'] = dga['label'].apply(lambda x : label_map[x])def convertStr2AscInt(x):    return [ord(c) for c in x]dga['name_v'] = dga['name'].apply(lambda x:convertStr2AscInt(x))dga.reset_index(inplace=True,drop=True)X = dga['name_v']Y = dga['label_v']Y= dga['b_label']X = pad_sequences(X, maxlen=75,dtype='float32')
```

预处理后的结果如下所示，如上图所示，白名单在左，黑名单在右，完全转换成一串数字了：

![图片](assets/7e00163cbe99.png)

#### LSTM 神经网络训练

训练的模型也很简单，将原始数据拆分为训练集和测试集

```
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)nb_classes  = len(set(Y))
```

构建神经网络

```
#[a-z]->26 [0-9]->10 [-_,]->3sess = tf.Session()  K.set_session(sess) max_features = 128model=Sequential()model.add(Embedding(max_features, 128,name="inputlayer"))model.add(LSTM(128))model.add(Dropout(0.5))model.add(Dense(128, kernel_initializer='uniform', activation='relu'))model.add(Dense(nb_classes, kernel_initializer='uniform', activation='softmax',name="outputlayer"))model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])model.summary()
```

由于特征还算明显，随意执行训练一下就好

```
dummy_y = np_utils.to_categorical(y_train)model.fit(X_train, dummy_y,batch_size=8192, epochs=50)
```

验证结果，准确度满足要求

```
dummy_test = np_utils.to_categorical(y_test)score = model.evaluate(X_test, dummy_test, batch_size=8192)score[0.0234935814024454, 0.9926283492534125]
```

获取模型的每层的名字，后续使用Go调用模型时会使用

```
[n.name for n in tf.get_default_graph().as_graph_def().node]'inputlayer_input', 'inputlayer/random_uniform/shape', 'inputlayer/random_uniform/min', 'inputlayer/random_uniform/max', ... 'inputlayer/Cast', 'inputlayer/embedding_lookup/axis', 'inputlayer/embedding_lookup', 'lstm_1/random_uniform/shape', 'lstm_1/random_uniform/min', 'lstm_1/random_uniform/max', ... 'outputlayer/MatMul', 'outputlayer/BiasAdd', 'outputlayer/Softmax',
```

最后导出模型

```
builder = tf.saved_model.builder.SavedModelBuilder("dga")  builder.add_meta_graph_and_variables(sess, ["cisco"])  builder.save()  #sess.close()  
```

#### 实时推理

利用`Tensorflow/go`实现,模型加载代码如下

```
type DGAModel struct { model *tf.SavedModel}func New(modelpath string) *DGAModel { model, err := tf.LoadSavedModel(modelpath, []string{"cisco"}, nil) if err != nil {  logrus.Fatal("Error loading saved model:", err.Error()) } return &DGAModel{  model: model, }}
```

原始从DNS获得的域名需要进行预处理

```
//SeqPadding : function to convert dns string to float32 array with left paddingfunc SeqPadding(dns string) [MAX_LEN]float32 { var X [MAX_LEN]float32 namestr := []byte(dns) strlen := len(namestr) if strlen >= MAX_LEN {  namestr = namestr[strlen-MAX_LEN : strlen]  for idx := 0; idx < MAX_LEN; idx++ {   X[idx] = float32(namestr[idx])   if X[idx] >= 128 {    X[idx] = 0   }  } } else {  idy := 0  for idx := MAX_LEN - strlen; idx < MAX_LEN; idx++ {   X[idx] = float32(namestr[idy])   if X[idx] >= 128 {    X[idx] = 0   }   idy++  } } return X}
```

模型预测推理,注意其中的`inputlayer_input` 和`outputlayer/Softmax`就是前一节中的提到的层

```
//Predict :main function to predict domain riskfunc (d *DGAModel) Predict(dns string) bool { //normally dynamic domain needs more than 5 chars. if len(dns) < 5 {  return true } X := SeqPadding(dns) tensors, _ := tf.NewTensor([][MAX_LEN]float32{X}) r, err := d.model.Session.Run(  map[tf.Output]*tf.Tensor{   d.model.Graph.Operation("inputlayer_input").Output(0): tensors,  },  []tf.Output{   d.model.Graph.Operation("outputlayer/Softmax").Output(0),  },  nil, ) if err != nil {  logrus.Fatal(err) } else {  rlist := r[0].Value().([][]float32)  if rlist[0][0] < 0.1 {   return true  } } return false}
```

最终我们就可以在DNS HandleFunc中调用过滤了：

```
dgaModel := dga.New("./model/dga") dns.HandleFunc(".", func(w dns.ResponseWriter, r *dns.Msg) {  if len(r.Question) == 0 {   return  }  question := r.Question[0]  //DGA Domain security check  isDGA := dgaModel.Predict(question.Name)  if isDGA {   logrus.Warn("DGA: ", question.Name)   resp := new(dns.Msg)   resp.SetReply(r)   w.WriteMsg(resp)   return  }  //proxy logic  result, err := p.GetResponse(r)  if err == nil {   w.WriteMsg(result)  } })
```

### 未来展望

ZaDNS以后会在时域上检测DNS请求的依赖关系，增加Whois的支持,增加Cloud Native的部署方式构建一个相对较大的集群，当然在某司内部也会将其用于SDWAN 和一些传统的Non-SDWAN路由器的广域网优化和调度算法研究，大概就这样了..欢迎大家来Github PR, 作为主机网络优化的一部分，这个项目后期也会整合到Ruta中.

Ruta的开源需要等研究完Torfino和Silicon One以及Xilinx FPGA Marvel Octeon 106xx后再决定下一版RFC-Draft的报文封装格式，因为很多事情都还是要迁就不同的硬件实现的.说到报文封装和处理，昨天看到华为数通发布世界级挑战课题， 转发算法实验室好好读读我以前写的那几篇`包处理的艺术`吧, 点到为止.