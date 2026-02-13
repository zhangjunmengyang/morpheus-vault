# RuntimeFilter

## RunTimeFilter

Doris做HashJoin需要先读取右表，生成哈希表，左表再流式的通过右表的哈希表从而得出 Join 结果。RunTimeFilter做的事情就是将右表的结果下推到左表进行过滤，加速左表的扫描。 也正因为这个原理，不支持left join、outer join等需要左表全部返回的SQL。 

当前 Doris 支持三种类型 RuntimeFilter

- IN：将一个 hashset 下推到数据扫描节点。In可以触发索引的过滤。
- BloomFilter：利用哈希表的数据构造一个 BloomFilter，然后把这个 BloomFilter 下推到查询数据的扫描节点。BloomFilter不能触发索引的过滤。
- MinMax：就是个 Range 范围，通过右表数据确定 Range 范围之后，下推给数据扫描节点。 
在子查询中，使用runtime_filter_type='bloom_filter'和runtime_filter_type='in'过滤器会对过滤的数据量级产生不同的影响。

- 当使用Bloom Filter时，它可以高效地确定一些显然不包含目标元素的块(Chunk)，将这些块进行跳过，从而明显减少需要扫描的块数和时间开销。但是，如果该块上实际存在目标元素，则仍然需要进行查询操作，因此可能会比较慢。
- 而当使用'in'运行时过滤器时，它可以在最开始的时候就将过滤条件与值集合进行比较，并从原始块中删除不匹配的行，从而明显减少需要扫描的数据量级和时间消耗。 但是，由于'in'过滤器需要完全扫描原始块，因此可能会增加资源消耗。
总的来说，在数据量很大且筛选结果可能较少的情况下，推荐使用Bloom Filter进行优化，而在数据量较小且需要严格匹配条件的情况下，推荐使用'in'过滤器。

**RuntimeFilter 是每个 instance 的右表？还是合并后的？**

- 支持Global模式，多个Instance右表产生的RuntimeFilter会合并
- 支持Local模式，单个Instance右表产生的RuntimeFilter只用于当前instance的左表
- 支持设置runtime filter的等待时间，等待时间内到达就可以下推到存储引擎过滤，等待超时后到达就需要通过将数据读取上来后再过滤