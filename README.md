# PrunedLandmarkLabeling

* Usage
  * Build Index
    * python pll.py build -m [map_file_name] -o [order_mode]
      * map_file_name: The map file downloaded by osmnx
      * order_mode: specify the way to build the vectice order
        * 0: test order, that means the vectices ordered Sequentially
        * 1: random order, 
        * 2: degree-based order,
        * 3: betweenness-based order,
        * 4: 2-hop-based order
    * Addition Feature
      * -m: use multi-thread.
  * Query distance
    * python pll.py query -s [src_vertex] -t [target_vectex]
      * src_vertex: the source vertex of the query
      * target_vectex: the target vertex of the query
  * Algorithm Validation
    * python pll.py test -t [times] -m [map_file]
      * times: specify the number of validation cases
      * map_file: the map_file of the built index

  * 说明：（以test2.map图为例）
    * 构建order: `python pll.py -i test2.map -o 3`   (这里的3是代表betweenness-based order)
    * 100000次查询： `pyhon test_query.py -i test2.map`  
    * 顺序问题： 
      * 需要先build后，此时ppl.idx文件写入了正确的index，才能进行查询
      * 2-hop-based order是对已有的Label进行构建的新的order，所以需要先采用其他三种order构建策略进行build后，才能构建2-hop-based order， eg： `python pll.py -i test2.map -o 3`后才能进行`python pll.py -i test2.map -o 4`