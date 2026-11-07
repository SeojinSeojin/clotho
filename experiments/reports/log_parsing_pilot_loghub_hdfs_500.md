# log_parsing pilot: loghub_hdfs_500 (llama, generator: gen_vllm)

500 inputs x 10 samples = 5000 outputs

| metric | value |
|---|---|
| output-level normalized match (Clotho label) | 10.6% |
| output-level strict match | 9.9% |
| output-level parse_failure | 0.0% |
| input-level mean test score | 0.106 |
| inputs labelled PASS (score > 0.5) | 2.6% |
| inputs with 10/10 pass / 0/10 pass | 0 / 303 |

## Runtime per shard

| stage | shard | inputs | seconds | s/input |
|---|---|---|---|---|
| gen_vllm | 0 | 250 | 25 | 0.10 |
| gen_vllm | 1 | 250 | 25 | 0.10 |
| hidden | 0 | 250 | 29 | 0.11 |
| hidden | 1 | 250 | 30 | 0.12 |

## Pass rate by template (10 hardest)

| template_id | n | mean score | template |
|---|---|---|---|
| E8 | 13 | 0.00 | BLOCK* NameSystem.addStoredBlock: addStoredBlock request received for &lt;*&gt; on &lt;*&gt; size &lt;*&gt; But it does not belong to any file. |
| E2 | 13 | 0.00 | Exception in receiveBlock for block &lt;*&gt; java.io.InterruptedIOException: Interruped while waiting for IO on channel java.nio.channels.SocketChannel[con … |
| E5 | 14 | 0.00 | PacketResponder &lt;*&gt; &lt;*&gt; Exception java.io.InterruptedIOException: Interruped while waiting for IO on channel java.nio.channels.SocketChannel[connected … |
| E33 | 14 | 0.00 | &lt;*&gt; Starting thread to transfer block &lt;*&gt; to &lt;*&gt;, &lt;*&gt; |
| E39 | 13 | 0.00 | Received block &lt;*&gt; src: &lt;*&gt; dest: &lt;*&gt; of size &lt;*&gt; |
| E35 | 13 | 0.00 | &lt;*&gt;:Exception writing block &lt;*&gt; to mirror &lt;*&gt; |
| E19 | 9 | 0.00 | &lt;*&gt;:Failed to transfer &lt;*&gt; to &lt;*&gt; got java.io.IOException: Connection reset by peer |
| E45 | 13 | 0.00 | &lt;*&gt; Served block &lt;*&gt; to &lt;*&gt; |
| E38 | 14 | 0.00 | &lt;*&gt;:Got exception while serving &lt;*&gt; to &lt;*&gt;: |
| E34 | 14 | 0.00 | BLOCK* NameSystem.allocateBlock: &lt;*&gt; &lt;*&gt; |

## 20 random failures (of 4468: {'mismatch': 4467, 'parse_failure': 1}; seed 0)

| # | id | input | ground truth | model output (parsed; raw if parse failure) | type |
|---|---|---|---|---|---|
| 1 | hdfs-3019087 | PacketResponder 0 for block blk_-8837062228229592478 terminating | PacketResponder &lt;*&gt; for block &lt;*&gt; terminating | PacketResponder &lt;*&gt; for block blk_-&lt;*&gt; terminating | mismatch |
| 2 | hdfs-8834418 | BLOCK* NameSystem.allocateBlock: /user/root/rand6/_temporary/_task_200811101024_0013_m_001092_0/part-01092. blk_3775819689072964469 | BLOCK* NameSystem.allocateBlock: &lt;*&gt; &lt;*&gt; | BLOCK* NameSystem.allocateBlock: /user/root/&lt;*&gt;/_temporary/_task_ &lt;*&gt; _ &lt;*&gt; _part-&lt;*&gt;. blk_&lt;*&gt; | mismatch |
| 3 | hdfs-4700154 | Received block blk_2741965161086243141 src: /10.250.10.6:43696 dest: /10.250.10.6:50010 of size 67108864 | Received block &lt;*&gt; src: &lt;*&gt; dest: &lt;*&gt; of size &lt;*&gt; | Received block blk_&lt;*&gt; src: /&lt;*&gt;:&lt;*&gt; dest: /&lt;*&gt;:&lt;*&gt; of size &lt;*&gt; | mismatch |
| 4 | hdfs-4673483 | writeBlock blk_7228855902021006130 received exception java.io.EOFException | writeBlock &lt;*&gt; received exception java.io.EOFException | writeBlock blk_&lt;*&gt; received exception java.io.&lt;?&gt;Exception | mismatch |
| 5 | hdfs-3048499 | BLOCK* NameSystem.allocateBlock: /user/root/randtxt2/_temporary/_task_200811101024_0002_m_000220_0/part-00220. blk_1630789733336799950 | BLOCK* NameSystem.allocateBlock: &lt;*&gt; &lt;*&gt; | BLOCK* NameSystem.allocateBlock: /&lt;*&gt;/&lt;*&gt;/&lt;*&gt;/&lt;*&gt;/&lt;*&gt;_part-&lt;*&gt;. blk_&lt;*&gt; | mismatch |
| 6 | hdfs-5753676 | Exception in receiveBlock for block blk_-539148231797001003 java.io.EOFException | Exception in receiveBlock for block &lt;*&gt; java.io.EOFException | Exception in receiveBlock for block &lt;*&gt; java.io.&lt;*&gt; | mismatch |
| 7 | hdfs-6525452 | PacketResponder blk_-5375146817488963214 2 Exception java.net.SocketTimeoutException: 120000 millis timeout while waiting for channel to be ready for read. ch : java.nio.channels.SocketChannel[connected local=/10.251.65.203:60847 remote=/10.251.30.85:50010] | PacketResponder &lt;*&gt; &lt;*&gt; Exception java.net.SocketTimeoutException: &lt;*&gt; millis timeout while waiting for channel to be ready for read. ch : java.nio.channels.SocketChannel[connected local=&lt;*&gt; remote=&lt;*&gt;] | PacketResponder &lt;*&gt; 2 Exception java.net.SocketTimeoutException: &lt;*&gt; millis timeout while waiting for channel to be ready for read. ch : java.nio.channels.SocketChannel[connected local=/&lt;*&gt; remote=/&lt;*&gt;] | mismatch |
| 8 | hdfs-2635625 | BLOCK* NameSystem.addStoredBlock: addStoredBlock request received for blk_-8280841389336725318 on 10.251.106.10:50010 size 67108864 But it does not belong to any file. | BLOCK* NameSystem.addStoredBlock: addStoredBlock request received for &lt;*&gt; on &lt;*&gt; size &lt;*&gt; But it does not belong to any file. | BLOCK* NameSystem.addStoredBlock: addStoredBlock request received for blk-&lt;**&gt; on &lt;*&gt;:&lt;**&gt; size &lt;**&gt; But it does not belong to any file. | mismatch |
| 9 | hdfs-855345 | Verification succeeded for blk_-5969169609767705593 | Verification succeeded for &lt;*&gt; | Verification succeeded for blk_&lt;*&gt; | mismatch |
| 10 | hdfs-8725100 | PacketResponder blk_377785190538046925 2 Exception java.net.SocketTimeoutException: 120000 millis timeout while waiting for channel to be ready for read. ch : java.nio.channels.SocketChannel[connected local=/10.251.67.211:59377 remote=/10.251.107.50:50010] | PacketResponder &lt;*&gt; &lt;*&gt; Exception java.net.SocketTimeoutException: &lt;*&gt; millis timeout while waiting for channel to be ready for read. ch : java.nio.channels.SocketChannel[connected local=&lt;*&gt; remote=&lt;*&gt;] | PacketResponder &lt;*&gt; &lt;*&gt; &lt;*&gt; Exception java.net.SocketTimeoutException: &lt;*&gt; millis timeout while waiting for channel to be ready for read. ch : java.nio.channels.SocketChannel[connected local=/&lt;&gt; remote=/&lt;&gt;] | mismatch |
| 11 | hdfs-4673446 | PacketResponder blk_7228855902021006130 1 Exception java.io.InterruptedIOException: Interruped while waiting for IO on channel java.nio.channels.SocketChannel[connected local=/10.251.123.33:59272 remote=/10.251.110.68:50010]. 59985 millis timeout left. | PacketResponder &lt;*&gt; &lt;*&gt; Exception java.io.InterruptedIOException: Interruped while waiting for IO on channel java.nio.channels.SocketChannel[connected local=&lt;*&gt; remote=&lt;*&gt;]. &lt;*&gt; millis timeout left. | PacketResponder blk_&lt;*&gt; 1 Exception java.io.InterruptedIOException: Interruped while waiting for IO on channel java.nio.channels.SocketChannel[connected local=/&lt;*&gt;:&lt;*&gt; remote=/&lt;*&gt;:&lt;*&gt;]. &lt;*&gt; millis timeout left. | mismatch |
| 12 | hdfs-7022284 | Adding an already existing block blk_-7052911997539087826 | Adding an already existing block &lt;*&gt; | Adding an already existing block blk-&lt;&gt; | mismatch |
| 13 | hdfs-10432792 | Receiving empty packet for block blk_-8895314640671734566 | Receiving empty packet for block &lt;*&gt; | Receiving empty packet for block blk_&lt;&gt; | mismatch |
| 14 | hdfs-6517045 | PacketResponder blk_-4380884313697280861 1 Exception java.net.SocketTimeoutException: 60000 millis timeout while waiting for channel to be ready for read. ch : java.nio.channels.SocketChannel[connected local=/10.251.107.50:40367 remote=/10.251.30.85:50010] | PacketResponder &lt;*&gt; &lt;*&gt; Exception java.net.SocketTimeoutException: &lt;*&gt; millis timeout while waiting for channel to be ready for read. ch : java.nio.channels.SocketChannel[connected local=&lt;*&gt; remote=&lt;*&gt;] | PacketResponder &lt;*&gt; &lt;*&gt; &lt;*&gt; Exception &lt;*&gt;: &lt;*&gt; ms timeout while waiting for channel to be ready for read. ch : java.nio.channels.SocketChannel[connected local=/&lt;*&gt;:&lt;*&gt; remote=/&lt;*&gt;:&lt;*&gt;] | mismatch |
| 15 | hdfs-585198 | 10.251.193.175:50010:Got exception while serving blk_-7576013425402387932 to /10.251.193.175: | &lt;*&gt;:Got exception while serving &lt;*&gt; to &lt;*&gt;: | &lt;&gt;:&lt;&gt;:Got exception while serving blk_-&lt;&gt; to /&lt;&gt;:. | mismatch |
| 16 | hdfs-1934169 | PacketResponder blk_5678483308337211471 1 Exception java.io.InterruptedIOException: Interruped while waiting for IO on channel java.nio.channels.SocketChannel[connected local=/10.251.38.214:60180 remote=/10.250.7.244:50010]. 30644 millis timeout left. | PacketResponder &lt;*&gt; &lt;*&gt; Exception java.io.InterruptedIOException: Interruped while waiting for IO on channel java.nio.channels.SocketChannel[connected local=&lt;*&gt; remote=&lt;*&gt;]. &lt;*&gt; millis timeout left. | PacketResponder blk_&lt;*&gt; 1 Exception java.io.InterruptedIOException: Interruped while waiting for IO on channel java.nio.channels.SocketChannel[connected local=/&lt;*&gt;:&lt;*&gt; remote=/&lt;*&gt;:&lt;*&gt;]. &lt;*&gt; millis timeout left. | mismatch |
| 17 | hdfs-139791 | PacketResponder blk_1528078116812077719 2 Exception java.io.EOFException | PacketResponder &lt;*&gt; &lt;*&gt; Exception java.io.EOFException | PacketResponder &lt;*&gt; &lt;*&gt; 2 Exception &lt;*&gt; | mismatch |
| 18 | hdfs-3630160 | Verification succeeded for blk_4392897483347213035 | Verification succeeded for &lt;*&gt; | Verification succeeded for blk_&lt;*&gt; | mismatch |
| 19 | hdfs-9405304 | Receiving empty packet for block blk_7381152960626506881 | Receiving empty packet for block &lt;*&gt; | Receiving empty packet for block blk_&lt;*&gt; | mismatch |
| 20 | hdfs-8713339 | Exception in receiveBlock for block blk_6224343649004202692 java.net.SocketTimeoutException: 490000 millis timeout while waiting for channel to be ready for write. ch : java.nio.channels.SocketChannel[connected local=/10.251.127.243:44092 remote=/10.251.107.50:50010] | Exception in receiveBlock for block &lt;*&gt; java.net.SocketTimeoutException: &lt;*&gt; millis timeout while waiting for channel to be ready for write. ch : java.nio.channels.SocketChannel[connected local=&lt;*&gt; remote=&lt;*&gt;] | Exception in receiveBlock for block blk_&lt;*&gt; java.net.SocketTimeoutException: &lt;*&gt; millis timeout while waiting for channel to be ready for write. ch : java.nio.channels.SocketChannel[connected local=/&lt;*&gt;:&lt;*&gt; remote=/&lt;*&gt;:&lt;*&gt; | mismatch |

## Clotho ROC-AUC (inputs not yet in the reference set)

**layer_21_seed_0 / random_pca**: first fitted iteration 24

| iteration | reference set | test inputs | test PASS | ROC-AUC |
|---|---|---|---|---|
| 24 | 240 | 270 | 4 | 0.779 |
| 25 | 250 | 260 | 3 | 0.958 |
| 26 | 260 | 250 | 3 | 0.961 |
| 27 | 270 | 240 | 3 | 0.966 |
| 28 | 280 | 230 | 3 | 0.965 |
| 29 | 290 | 220 | 3 | 0.934 |
| 30 | 300 | 210 | 3 | 0.931 |
| 31 | 310 | 200 | 3 | 0.958 |
| 32 | 320 | 190 | 3 | 0.954 |
| 33 | 330 | 180 | 3 | 0.959 |
| 34 | 340 | 170 | 3 | 0.952 |
| 35 | 350 | 160 | 3 | 0.958 |
| 36 | 360 | 150 | 3 | 0.952 |
| 37 | 370 | 140 | 2 | 0.989 |
| 38 | 380 | 130 | 2 | 0.992 |
| 39 | 390 | 120 | 2 | 0.992 |
| 40 | 400 | 110 | 1 | 0.991 |
| 41 | 410 | 100 | 1 | 1.000 |
| 42 | 420 | 90 | 1 | 1.000 |
| 43 | 430 | 80 | 1 | 1.000 |
| 44 | 440 | 70 | 0 | nan |
| 45 | 450 | 60 | 0 | nan |
| 46 | 460 | 50 | 0 | nan |
| 47 | 470 | 40 | 0 | nan |
| 48 | 480 | 30 | 0 | nan |
| 49 | 490 | 20 | 0 | nan |
| 50 | 500 | 10 | 0 | nan |
