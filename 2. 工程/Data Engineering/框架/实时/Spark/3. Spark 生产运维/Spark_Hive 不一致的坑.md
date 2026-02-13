# Spark/Hive 不一致的坑

## 一、可接受的不一致

### 1.1、随机算子

**时间函数**

- unixtimestamp：产生当前系统时间戳
Flink sql 里面的 localtimestamp 也涉及这种问题，可能影响回溯

**窗口函数**

- **row_number**：通常用法是按照某列分组，在组内按另一列排序，并给排序列一个排位数。
- **lag**:在窗口中，当前行的前n行的值。
- **lead**:在窗口中，当前行的后n行的值
```
-- 举例
case 
    when c=1 AND LEAD(a,1) OVER(ORDER BY a,b)<>a then 1
    when c=1 AND LEAD(a,1) OVER(ORDER BY a,b)=a AND LEAD(c,1) OVER(ORDER BY a,b)=1 then -1 
    else 0
end as accept
```

如果数据回溯过程回放顺序不同，就可能导致不同的结果（很有可能，平台在线测试也只是大致的时间范围，无法完全还原现场）

### 1.2、数字精度

1. 类型间转换
1. 物理计划中被计算的数据在任务中划分不同，主要在于小数运算不严格满足结合律，可能不一致
1. 对round函数产生的结果再进行转换后，不能保留原小数精度。
```
select cast(round(cast(0.333333 as float), 4) as double);
```

**Spark的结果是0.33329999446868896，而Hive是0.3333。**

很多时候，这类错误是由于隐式转换造成的，比如用insert overwrite table tableA round(cast(0.333333 as float), 4); 

tableA中对应的列是一个double类型，Spark 会默认增加一个转换到double的隐式转换，就像上面的示例SQL，但由于丢失了精度，因此会产生和 Hive 不一致的问题。

### 1.3、算子实现算法

percentile_approx、hash、一些数学计算等函数由于实现算法不同，有可能结果不完全一致

```
select sum('');  -- Spark: NULL   Hive:0.0
```

### 1.4、对数据理解不一致

比如 map 中 key 能否为 null

```
select map(null, null); -- Spark: 报错   Hive: {null: null}
```

## 二、不可接受的不一致

### 2.1、隐式转换规则不一致

1. 将bigint转换成timestamp类型时，Spark默认bigint是一个秒数，而Hive默认是bigint是一个毫秒数
1. string 类型和 bigint 类型比较的精度损失问题
### 2.2、写法不规范导致

比如 Spark将算子的两个参数转换成date类型处理，从而识别范围是0001-01-01到9999-12-31,但Hive的不同，使用了SimpleDateFormat直接对参数进行处理，有些用户会写出一些比较不常用的写法，如datediff('2015-10-09', '0-0-0')；在Spark中会返回NULL，而Hive得到736279。
