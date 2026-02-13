# Flink SideOutput

**旁路分流器**：按照频道分流

三种方法：

1. filter：true 的保留，false 丢弃
1. 优点：简单
1. 缺点：需要多次遍历，效率低
1. split：在算子中定义 outputselector，然后重写 select 方法，对不同数据标记，最后分出来
1. 缺点：不能多次 split，会报错，解决是用 sideoutput
1. sideoutput：支持多次分流，使用时先定义 outputtag，然后使用函数拆分，比如 keyedprocessfunction