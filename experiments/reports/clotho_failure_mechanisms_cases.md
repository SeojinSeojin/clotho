# Systematically sampled cases: assistant qualitative inspection

All112 selected cases were inspected by the assistant. These are exploratory language judgments, not independent expert semantic labels. Selection/outcomes use held-out template-hash evaluation halves and seed0. Risk is negative PASS log density; percentiles refer to the full original job test population. First/last votes are strict successes out of five; sample0 is not the target label. Three nearest real references are raw-hidden cosine neighbors. Full packets and matching records preserve original identifiers.


## C001: llama/standard/Zookeeper/FN

ID `zookeeper-26866`, index459, seed0, slotextreme; seen GT=True. Risk-95.568595, threshold-6.667789, percentile19.39%, margin/IQR-1.3669. First/last votes5/5,2/5; failure label=1. Other method percentiles: Clotho=19.4%, lexical_LR=27.2%, raw_hidden_LR=24.3%, same_space_LR=11.8%, same_space_LR_real_only=9.1%, same_space_two_density=1.8%, structural_LR=2.2%

Raw log:
```text
******* GOODBYE /10.10.34.11:52407 ********
```
GT:
```text
******* GOODBYE <*> ********
```
Sample0:
```text
******* GOODBYE <*> ********
```

Target-batch parsed outputs:
- Sample5: `******* GOODBYE <*> ********`
- Sample6: `********* GOODBYE <*> *********`
- Sample7: `********** GOODBYE <*> ********`
- Sample8: `********* GOODBYE <*> *********`
- Sample9: `******* GOODBYE <*> ********`

Nearest labeled references:
- Index92, FAIL=False: `******* GOODBYE /10.10.34.11:52690 ********`; GT=`******* GOODBYE <*> ********`
- Index123, FAIL=False: `Received connection request /10.10.34.11:46101`; GT=`Received connection request <*>`
- Index422, FAIL=False: `Received connection request /10.10.34.11:58842`; GT=`Received connection request <*>`

Same-family successful neighbor and first-batch 5/5 success fit low risk; last-batch star-count changes reverse the label. H3 counterexample to interpreting every FN as missing a stable mechanism; compare stable FN C005.


## C002: llama/standard/Zookeeper/FN

ID `zookeeper-52248`, index489, seed0, slotextreme; seen GT=True. Risk-94.022532, threshold-6.667789, percentile21.83%, margin/IQR-1.3431. First/last votes1/5,2/5; failure label=1. Other method percentiles: Clotho=21.8%, lexical_LR=67.0%, raw_hidden_LR=75.7%, same_space_LR=55.7%, same_space_LR_real_only=62.8%, same_space_two_density=60.6%, structural_LR=70.6%

Raw log:
```text
Server environment:host.name=node3
```
GT:
```text
Server environment:host.name=<*>
```
Sample0:
```text
environment:host.name=<*>,
```

Target-batch parsed outputs:
- Sample5: `Server environment:host.<*>`
- Sample6: `Server environment:host.name=<*>`
- Sample7: `Server environment:host.name=<*>`
- Sample8: `Server environment: host.name=<*>`
- Sample9: `Server environment:host.<*>`

Nearest labeled references:
- Index242, FAIL=False: `Server environment:host.name=node2`; GT=`Server environment:host.name=<*>`
- Index388, FAIL=True: `Server environment:host.name=node1`; GT=`Server environment:host.name=<*>`
- Index270, FAIL=True: `Server environment:os.name=Linux`; GT=`Server environment:os.name=<*>`

Host-name family is familiar, but deleting name or changing spacing produces unstable correctness. A successful same-family neighbor can support low density risk without resolving boundary conventions; compare C083.


## C003: llama/standard/Spark/FN

ID `spark-2662447`, index414, seed0, slotnear_threshold; seen GT=False. Risk74.379848, threshold74.455715, percentile48.17%, margin/IQR-0.0006. First/last votes0/5,1/5; failure label=1. Other method percentiles: Clotho=48.2%, lexical_LR=43.5%, raw_hidden_LR=50.6%, same_space_LR=70.1%, same_space_LR_real_only=62.8%, same_space_two_density=44.5%, structural_LR=80.4%

Raw log:
```text
Still have 1 requests outstanding when connection from mesos-slave-07/10.10.34.17:54953 is closed
```
GT:
```text
Still have <*> requests outstanding when connection from <*> is closed
```
Sample0:
```text
Still have <*> requests outstanding when connection from mesos-slave-07/<*>:<*> is closed
```

Target-batch parsed outputs:
- Sample5: `Still have <*>, outstanding when connection from <*> is closed`
- Sample6: `Still have <*> requests outstanding when connection from <*>/<*>:<*> is closed`
- Sample7: `Still have 1 requests outstanding when connection from <*>:54953 is closed`
- Sample8: `Still have 1 requests outstanding when connection from <*> is closed`
- Sample9: `Still have <*> requests outstanding when connection from <*> is closed`

Nearest labeled references:
- Index101, FAIL=False: `Lost executor 4 on mesos-slave-11: Executor heartbeat timed out after 147429 ms`; GT=`Lost executor <*> on <*>: Executor heartbeat timed out after <*> ms`
- Index333, FAIL=False: `Lost executor 9 on mesos-slave-06: Executor heartbeat timed out after 154834 ms`; GT=`Lost executor <*> on <*>: Executor heartbeat timed out after <*> ms`
- Index22, FAIL=True: `Remoting started; listening on addresses :[akka.tcp://sparkExecutorActorSystem@mesos-slave-28:56300]`; GT=`Remoting started; listening on addresses :[<*>]`

Host/IP/port boundary variants mostly fail despite a successful heartbeat neighbor. The near-zero margin gives weak evidence of a confident miss; compare consistently correct numeric boundary case C026.


## C004: llama/standard/HealthApp/FN

ID `healthapp-196605`, index114, seed0, slotnear_threshold; seen GT=False. Risk19.342956, threshold19.656123, percentile46.95%, margin/IQR-0.0022. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=47.0%, lexical_LR=67.9%, raw_hidden_LR=23.8%, same_space_LR=49.9%, same_space_LR_real_only=49.6%, same_space_two_density=56.7%, structural_LR=73.3%

Raw log:
```text
aggregateHiHealthData() time = 1514890379521, totalTime = 35
```
GT:
```text
aggregateHiHealthData() time = <*> totalTime = <*>
```
Sample0:
```text
aggregateHiHealthData() time = 1514890379521, totalTime = 35
```

Target-batch parsed outputs:
- Sample5: `aggregate<*>Data() time = 1514890379521, totalTime = 35`
- Sample6: `aggregateHiHealthData() time = <>, totalTime = <>`
- Sample7: `aggregateHiHealthData() time = 1514890379521, totalTime = 35`
- Sample8: `aggregateHiHealthData() time = 1514890379521, totalTime = 35`
- Sample9: `aggregateHiHealthData() time = <*>, totalTime = <*>`

Nearest labeled references:
- Index147, FAIL=False: `saveRealTimeHealthDatasStat() size = 1,totalTime = 29`; GT=`saveRealTimeHealthDatasStat() size = <*>,totalTime = <*>`
- Index181, FAIL=True: `bulkSaveDetailHiHealthData() size = 28,totalTime = 23`; GT=`bulkSaveDetailHiHealthData() size = <*>,totalTime = <*>`
- Index419, FAIL=False: `saveRealTimeHealthDatasStat() size = 1,totalTime = 23`; GT=`saveRealTimeHealthDatasStat() size = <*>,totalTime = <*>`

All ten fail, including copying numeric values and comma retention. A successful health-data neighbor does not transfer its boundary convention; near-threshold FN is not solely label noise. Compare C056.


## C005: llama/standard/HealthApp/FN

ID `healthapp-149633`, index160, seed0, slotseeded_representative; seen GT=True. Risk1.264379, threshold19.656123, percentile36.71%, margin/IQR-0.1273. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=36.7%, lexical_LR=77.9%, raw_hidden_LR=20.6%, same_space_LR=46.2%, same_space_LR_real_only=42.6%, same_space_two_density=64.8%, structural_LR=55.9%

Raw log:
```text
registerLogCallback com.huawei.health.c.c@8e48250
```
GT:
```text
registerLogCallback <*>
```
Sample0:
```text
registerLogCallback <*>@<*>.
```

Target-batch parsed outputs:
- Sample5: `registerLogCallback com.huawei.health.c.c@<*>`
- Sample6: `registerLogCallback com.huawei.<*>@<>`
- Sample7: `registerLogCallback com.huawei.health.c.c@<>`
- Sample8: `registerLogCallback com.huawei.health.c.c@<*>`
- Sample9: `registerLogCallback com.huawei.health.c.c@<*>`

Nearest labeled references:
- Index399, FAIL=True: `registerLogCallback com.huawei.health.c.c@d990a94`; GT=`registerLogCallback <*>`
- Index364, FAIL=False: `context=com.huawei.health.manager.DaemonService@191aeab`; GT=`context=<*>`
- Index239, FAIL=False: `processHandleBroadcastAction action:com.huawei.health.WIDGET_DISABLE`; GT=`processHandleBroadcastAction action:<*>`

All ten fail on an object-like word variable and a same-family failing neighbor exists, yet risk is low. Familiarity is insufficient and PASS-density scoring can disagree with local failure evidence; compare correctly high-risk C019.


## C006: llama/standard/Linux/FN

ID `linux-20027`, index93, seed0, slotseeded_representative; seen GT=False. Risk20.330114, threshold89.687182, percentile11.34%, margin/IQR-0.5868. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=11.3%, lexical_LR=59.9%, raw_hidden_LR=6.2%, same_space_LR=9.6%, same_space_LR_real_only=11.6%, same_space_two_density=22.1%, structural_LR=9.3%

Raw log:
```text
named shutdown failed
```
GT:
```text
<*> shutdown failed
```
Sample0:
```text
named <*> failed
```

Target-batch parsed outputs:
- Sample5: `named <*> failed`
- Sample6: `named <*> failed`
- Sample7: `named <*shutdown*> failed`
- Sample8: `named <*>, shutdown failed`
- Sample9: `named <*> failed`

Nearest labeled references:
- Index302, FAIL=False: `Kerberos authentication failed`; GT=`Kerberos authentication failed`
- Index438, FAIL=True: `ntpd shutdown succeeded`; GT=`<*> shutdown succeeded`
- Index86, FAIL=True: `crond shutdown succeeded`; GT=`<*> shutdown succeeded`

All ten fail by masking shutdown rather than named. Low score near a successful authentication phrase illustrates that general short-message similarity need not encode which word is variable; compare high-risk short failures C015.


## C007: llama/standard/Mac/FN

ID `mac-8721`, index286, seed0, slotseeded_representative; seen GT=False. Risk239.098849, threshold298.351914, percentile58.66%, margin/IQR-0.2228. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=58.7%, lexical_LR=50.6%, raw_hidden_LR=80.9%, same_space_LR=88.9%, same_space_LR_real_only=92.3%, same_space_two_density=61.1%, structural_LR=58.9%

Raw log:
```text
[HID] [ATC] [Error] AppleDeviceManagementHIDEventService::start Could not make a string from out poweredoff notification key
```
GT:
```text
[HID] [ATC] [Error] AppleDeviceManagementHIDEventService::start Could not make a string from out <*> notification key
```
Sample0:
```text
[HID] [ATC] [Error] AppleDeviceManagementHIDEventService::<*>::start Could not make a string from out poweredoff notification <*>
```

Target-batch parsed outputs:
- Sample5: `[HID] [ATC] [Error] AppleDeviceManagementHIDEventService::start Could not make a string from out <*> notification <*>`
- Sample6: `[HID] [ATC] [Error] AppleDeviceManagementHIDEventService::start Could not make a string from out <*> key`
- Sample7: `[HID] [ATC] [Error] AppleDeviceManagementHIDEventService::<*>Error> Could not make a string from out <*>`
- Sample8: `[HID] [ATC] [Error] <*>::start Could not make a string from out <*> notification key`
- Sample9: `[HID] [ATC] [Error] AppleDeviceManagementHIDEventService::<*>::start Could not make a string from out <*> notification <*> key`

Nearest labeled references:
- Index205, FAIL=True: `Metadata.framework [Error]: void _MDItemMarkAsUsedForPath(CFStringRef): was called with a NULL path`; GT=`Metadata.framework [Error]: void _MDItemMarkAsUsedForPath(<*>): was called with a <*> path`
- Index119, FAIL=True: `[10:29:35.631] <<<< CRABS >>>> crabsFlumeHostUnavailable: [0x7f961cf08cf0] Byte flume reports host unavailable.`; GT=`[<*>] <<<< CRABS >>>> crabsFlumeHostUnavailable: [<*>] Byte flume reports host unavailable.`
- Index297, FAIL=True: `CoreAnimation: warning, deleted thread with uncommitted CATransaction; set CA_DEBUG_TRANSACTIONS=1 in environment to log backtraces.`; GT=`CoreAnimation: warning, deleted thread with uncommitted CATransaction; set CA_DEBUG_TRANSACTIONS=<*> in environment to log backtraces.`

All ten fail on word-like poweredoff or constant class structure. A failing error-message neighbor does not ensure high-enough risk; near-system-relative high rank still falls below the learned threshold. Compare C104.


## C008: llama/standard/Apache/FP

ID `apache-9290`, index25, seed0, slotextreme; seen GT=True. Risk-2.170792, threshold-34.308093, percentile99.88%, margin/IQR14.3765. First/last votes3/5,4/5; failure label=0. Other method percentiles: Clotho=99.9%, lexical_LR=80.4%, raw_hidden_LR=99.4%, same_space_LR=98.9%, same_space_LR_real_only=93.3%, same_space_two_density=99.9%, structural_LR=39.4%

Raw log:
```text
[client 220.93.65.11] File does not exist: /var/www/html/scripts/..\xc1\x1c..
```
GT:
```text
[client <*>] File does not exist: <*>
```
Sample0:
```text
[client <*>] File does not exist: /<*>
```

Target-batch parsed outputs:
- Sample5: `[client <*>] File does not exist: <*>`
- Sample6: `[client <*>] File does not exist: <*>`
- Sample7: `[client <*>] File does not exist: <*>`
- Sample8: `[client <*>] File does not exist: <*>`
- Sample9: `[client <*>] File does not exist: /<*>`

Nearest labeled references:
- Index48, FAIL=True: `[client 220.176.190.155] File does not exist: /var/www/html/scripts/..\xc1\x1c..`; GT=`[client <*>] File does not exist: <*>`
- Index8, FAIL=True: `[client 61.185.32.30] File does not exist: /var/www/html/scripts/..\xc1\x9c..`; GT=`[client <*>] File does not exist: <*>`
- Index58, FAIL=True: `[client 61.151.191.3] File does not exist: /var/www/html/scripts/..%5c..`; GT=`[client <*>] File does not exist: <*>`

Escaped path looks unusual and the closest family example fails, but this line passes 4/5 in the label batch. High risk may reflect rare path form and unstable family outcome, not incapacity to parse paths; compare C028.


## C009: llama/standard/Apache/FP

ID `apache-3180`, index486, seed0, slotextreme; seen GT=True. Risk-11.507703, threshold-34.308093, percentile97.68%, margin/IQR10.1997. First/last votes2/5,4/5; failure label=0. Other method percentiles: Clotho=97.7%, lexical_LR=93.7%, raw_hidden_LR=96.7%, same_space_LR=98.4%, same_space_LR_real_only=98.9%, same_space_two_density=94.0%, structural_LR=36.7%

Raw log:
```text
mod_jk child workerEnv in error state 7
```
GT:
```text
mod_jk child workerEnv in error state <*>
```
Sample0:
```text
mod_jk child <*> in error state <*>
```

Target-batch parsed outputs:
- Sample5: `mod_jk child workerEnv in error state <*>`
- Sample6: `mod_jk child workerEnv in error state <*>`
- Sample7: `mod_jk child workerEnv in error state <*>`
- Sample8: `mod_jk child workerEnv in error state <*>`
- Sample9: `mod_jk <*> child <*> in error <*>`

Nearest labeled references:
- Index176, FAIL=True: `mod_jk child workerEnv in error state 6`; GT=`mod_jk child workerEnv in error state <*>`
- Index461, FAIL=True: `mod_jk child workerEnv in error state 1`; GT=`mod_jk child workerEnv in error state <*>`
- Index189, FAIL=False: `jk2_init() Found child 7906 in scoreboard slot 9`; GT=`jk2_init() Found child <*> in scoreboard slot <*>`

Closest numeric-state family example fails, while this line passes 4/5. Family-level failure concentration can generate FP even for a seen template; compare stable TP C019.


## C010: llama/standard/HPC/FP

ID `hpc-146157`, index34, seed0, slotnear_threshold; seen GT=True. Risk4.204276, threshold4.162231, percentile38.41%, margin/IQR0.0022. First/last votes4/5,5/5; failure label=0. Other method percentiles: Clotho=38.4%, lexical_LR=13.2%, raw_hidden_LR=2.8%, same_space_LR=2.1%, same_space_LR_real_only=0.6%, same_space_two_density=2.1%, structural_LR=87.3%

Raw log:
```text
ambient=38
```
GT:
```text
ambient=<*>
```
Sample0:
```text
ambient=<*>
```

Target-batch parsed outputs:
- Sample5: `ambient=<*>`
- Sample6: `ambient=<*>`
- Sample7: `ambient=<*>`
- Sample8: `ambient=<*>`
- Sample9: `ambient=<*>`

Nearest labeled references:
- Index444, FAIL=False: `ambient=34`; GT=`ambient=<*>`
- Index38, FAIL=False: `Failed subcommands 3406`; GT=`Failed subcommands <*>`
- Index209, FAIL=True: `ambient=40 threshold exceeded`; GT=`ambient=<*>`

Simple ambient assignment and successful same-template neighbor agree with success; the score barely crosses the calibration threshold. This FP mostly illustrates threshold sensitivity, not a strong unfamiliarity effect.


## C011: llama/standard/BGL/FP

ID `bgl-1065947`, index288, seed0, slotnear_threshold; seen GT=False. Risk28.198294, threshold27.033638, percentile24.51%, margin/IQR0.0127. First/last votes5/5,5/5; failure label=0. Other method percentiles: Clotho=24.5%, lexical_LR=67.1%, raw_hidden_LR=22.8%, same_space_LR=18.4%, same_space_LR_real_only=12.6%, same_space_two_density=20.6%, structural_LR=15.7%

Raw log:
```text
machine check status register: 0xa0800000
```
GT:
```text
machine check status register: <*>
```
Sample0:
```text
machine check status register: <*>
```

Target-batch parsed outputs:
- Sample5: `machine check status register: <*>`
- Sample6: `machine check status register: <*>`
- Sample7: `machine check status register: <*>`
- Sample8: `machine check status register: <*>`
- Sample9: `machine check status register: <*>`

Nearest labeled references:
- Index279, FAIL=False: `Floating Point Status and Control Register: 0x82004000`; GT=`Floating Point Status and Control Register: <*>`
- Index428, FAIL=True: `CE sym 1, at 0x152f9800, mask 0x04`; GT=`CE sym <*>, at <*>, mask <*>`
- Index55, FAIL=True: `generating core.3254`; GT=`generating core.<*>`

All ten correctly mask a hexadecimal value, with another successful register neighbor. Near-threshold FP counters a universal hexadecimal-variable difficulty explanation; compare BGL stable failure C021.


## C012: llama/standard/HealthApp/FP

ID `healthapp-149627`, index303, seed0, slotseeded_representative; seen GT=False. Risk109.182054, threshold19.656123, percentile73.29%, margin/IQR0.6194. First/last votes1/5,3/5; failure label=0. Other method percentiles: Clotho=73.3%, lexical_LR=47.0%, raw_hidden_LR=45.0%, same_space_LR=44.5%, same_space_LR_real_only=44.0%, same_space_two_density=69.9%, structural_LR=45.7%

Raw log:
```text
getMidWareCache mRemote=com.huawei.health.c.c@8e48250
```
GT:
```text
getMidWareCache mRemote=<*>
```
Sample0:
```text
getMidWareCache mRemote=<*>
```

Target-batch parsed outputs:
- Sample5: `getMidWareCache mRemote=<*>@<>`
- Sample6: `getMidWareCache mRemote=<*>`
- Sample7: `getMidWareCache mRemote=com.huawei.health.c.c@<*>`
- Sample8: `getMidWareCache mRemote=<*>`
- Sample9: `getMidWareCache mRemote=<*>`

Nearest labeled references:
- Index153, FAIL=True: `getTodaySportData mStepsRecordManager=com.huawei.health.g.c@8a2a501`; GT=`getTodaySportData mStepsRecordManager=<*>`
- Index399, FAIL=True: `registerLogCallback com.huawei.health.c.c@d990a94`; GT=`registerLogCallback <*>`
- Index265, FAIL=True: `startStepCounterInter mStandStepCounter = com.huawei.health.b.b.a@9252368`; GT=`startStepCounterInter mStandStepCounter = <*>`

Object-reference boundaries vary across generations; 1/5 versus 3/5 successes flip majority. A nearby failing object-field family makes high risk understandable but not correct for this batch; compare C005.


## C013: llama/standard/Hadoop/FP

ID `hadoop-22827`, index340, seed0, slotseeded_representative; seen GT=False. Risk108.005549, threshold82.451607, percentile55.49%, margin/IQR0.1775. First/last votes3/5,3/5; failure label=0. Other method percentiles: Clotho=55.5%, lexical_LR=76.7%, raw_hidden_LR=27.2%, same_space_LR=50.6%, same_space_LR_real_only=55.7%, same_space_two_density=60.9%, structural_LR=45.0%

Raw log:
```text
Failed to renew lease for [DFSClient_NONMAPREDUCE_1537864556_1] for 53 seconds. Will retry shortly ...
```
GT:
```text
Failed to renew lease for <*> for <*> seconds. Will retry shortly ...
```
Sample0:
```text
Failed to renew lease for <*> for <*> seconds. Will retry shortly ...
```

Target-batch parsed outputs:
- Sample5: `Failed to renew lease for <*> for <*> seconds. Will retry shortly ...`
- Sample6: `Failed to renew lease for <*> for <*> seconds. Will retry shortly ...`
- Sample7: `Failed to renew lease for <*> for 53 seconds. Will retry shortly ...`
- Sample8: `Failed to renew lease for <*> for 53 seconds. Will retry shortly ...`
- Sample9: `Failed to renew lease for <*> for <*> seconds. Will retry shortly ...`

Nearest labeled references:
- Index68, FAIL=True: `Failed to connect to MININT-FNANLI5.fareast.corp.microsoft.com:13562 with 1 map outputs`; GT=`Failed to connect to <*> with <*> map outputs`
- Index88, FAIL=False: `Diagnostics report from attempt_1445087491445_0004_m_000009_1001: Container killed by the ApplicationMaster.`; GT=`Diagnostics report from <*>: Container killed by the ApplicationMaster.`
- Index168, FAIL=True: `attempt_1445062781478_0015_m_000005_0 TaskAttempt Transitioned from ASSIGNED to RUNNING`; GT=`<*> TaskAttempt Transitioned from <*> to <*>`

Lease duration is sometimes retained, and both batches are 3/5 successful. Risk is near the threshold; a failing connection neighbor is only a loose structural match. Compare stable numeric success C026.


## C014: llama/standard/Zookeeper/FP

ID `zookeeper-26445`, index291, seed0, slotseeded_representative; seen GT=True. Risk26.375870, threshold-6.667789, percentile69.88%, margin/IQR0.5081. First/last votes4/5,4/5; failure label=0. Other method percentiles: Clotho=69.9%, lexical_LR=59.1%, raw_hidden_LR=20.6%, same_space_LR=71.3%, same_space_LR_real_only=66.7%, same_space_two_density=70.1%, structural_LR=52.3%

Raw log:
```text
Snapshotting: 0xd0000001b to /var/lib/zookeeper/version-2/snapshot.d0000001b
```
GT:
```text
Snapshotting: <*> to <*>
```
Sample0:
```text
Snapshotting: <*> to <*>
```

Target-batch parsed outputs:
- Sample5: `Snapshotting: <*> to <*>`
- Sample6: `Snapshotting: <*> to <*>`
- Sample7: `Snapshotting: <*> to <*>/<*>/snapshot.d<*>*.`
- Sample8: `Snapshotting: <*> to <*>`
- Sample9: `Snapshotting: <*> to <*>`

Nearest labeled references:
- Index290, FAIL=True: `Snapshotting: 0xb00000084 to /var/lib/zookeeper/version-2/snapshot.b00000084`; GT=`Snapshotting: <*> to <*>`
- Index253, FAIL=False: `Reading snapshot /var/lib/zookeeper/version-2/snapshot.d0000001b`; GT=`Reading snapshot <*>`
- Index439, FAIL=True: `Expiring session 0x34ed9ac1c1e0040, timeout of 10000ms exceeded`; GT=`Expiring session <*> timeout of <*> exceeded`

Same-family failing snapshot reference coexists with 4/5 successes in both batches. Risk can inherit family difficulty despite mostly correct path abstraction; compare TP C020 from this family.


## C015: llama/standard/Linux/TP

ID `linux-5371`, index319, seed0, slotextreme; seen GT=False. Risk7369.283954, threshold89.687182, percentile99.15%, margin/IQR61.5898. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=99.1%, lexical_LR=10.6%, raw_hidden_LR=74.5%, same_space_LR=62.6%, same_space_LR_real_only=51.3%, same_space_two_density=99.1%, structural_LR=4.3%

Raw log:
```text
Stopping squid:
```
GT:
```text
Stopping squid:
```
Sample0:
```text
Stopping squid: <*> is going down after <*> is terminated.
```

Target-batch parsed outputs:
- Sample5: `Stopping <*>:.`
- Sample6: `None`
- Sample7: `Jul 18 05:08:10  <*> squid[<*>]: squid[<*>] process received <*> signal.`
- Sample8: `Stopping squid`
- Sample9: `Stopping squid: <*> has crashed on <*> with error: <*> failed to connect to <*> on '*' in <*>s: <*>`

Nearest labeled references:
- Index83, FAIL=True: `*** info [client.c(284)]:`; GT=`*** info [client.c(<*>)]:`
- Index10, FAIL=True: `BIOS-provided physical RAM map:`; GT=`BIOS-provided physical RAM map:`
- Index236, FAIL=True: `Linux Plug and Play Support v0.97 (c) Adam Belay`; GT=`Linux Plug and Play Support <*> (c) Adam Belay`

Variable-free service header triggers invented continuation and malformed outputs in all ten. Very high risk and a failing short-header neighbor fit novelty-related success of prediction; compare successful short header C037.


## C016: llama/standard/Linux/TP

ID `linux-218`, index35, seed0, slotextreme; seen GT=False. Risk6077.669660, threshold89.687182, percentile98.66%, margin/IQR50.6620. First/last votes0/5,1/5; failure label=1. Other method percentiles: Clotho=98.7%, lexical_LR=19.6%, raw_hidden_LR=83.0%, same_space_LR=76.0%, same_space_LR_real_only=73.0%, same_space_two_density=98.7%, structural_LR=5.2%

Raw log:
```text
Starting background readahead:
```
GT:
```text
Starting background readahead:
```
Sample0:
```text
Starting <*>:
```

Target-batch parsed outputs:
- Sample5: `None`
- Sample6: `Starting background readahead: req_8812_03 to 192.168.0.5`
- Sample7: `Starting background readahead:`
- Sample8: `Starting background readahead: <*>.`
- Sample9: `Starting background readahead: by/default/data/req_8812_03 for /192.168.0.7:8080 on worker-3`

Nearest labeled references:
- Index10, FAIL=True: `BIOS-provided physical RAM map:`; GT=`BIOS-provided physical RAM map:`
- Index83, FAIL=True: `*** info [client.c(284)]:`; GT=`*** info [client.c(<*>)]:`
- Index175, FAIL=True: `checking if image is initramfs...it isn't (no cpio magic); looks like an initrd`; GT=`checking if image is initramfs...it isn't (no cpio magic); looks like an initrd`

A second variable-free header often triggers continuation; only one label-batch generation succeeds. This TP supports short-header risk but C037/C093 show that shortness is not sufficient for failure.


## C017: llama/standard/Mac/TP

ID `mac-9884`, index153, seed0, slotnear_threshold; seen GT=False. Risk298.391823, threshold298.351914, percentile66.71%, margin/IQR0.0002. First/last votes2/5,0/5; failure label=1. Other method percentiles: Clotho=66.7%, lexical_LR=36.2%, raw_hidden_LR=21.6%, same_space_LR=13.5%, same_space_LR_real_only=16.2%, same_space_two_density=71.6%, structural_LR=69.9%

Raw log:
```text
Page bounds {{0, 0}, {400, 400}}
```
GT:
```text
Page bounds {{<*>, <*>}, {<*>, <*>}}
```
Sample0:
```text
Page bounds {{<*>}, {<*>}}
```

Target-batch parsed outputs:
- Sample5: `Page bounds {{<*>}, {<*>}}`
- Sample6: `Page bounds {{<*>, <*>}`
- Sample7: `Page bounds {{<*>}, {<*>}}`
- Sample8: `Page bounds {{<*>}, {<*>}}`
- Sample9: `Page bounds {{<*>, <*>}, <*, *>}.`

Nearest labeled references:
- Index2, FAIL=True: `Arranged view frame: {{261, 0}, {450, 877}}`; GT=`Arranged view frame: {{<*>, <*>}, {<*>, <*>}}`
- Index393, FAIL=False: `Captive: CNPluginHandler en0: Inactive`; GT=`Captive: CNPluginHandler <*>: Inactive`
- Index165, FAIL=True: `00:04:54.559 : Terminate AirDrop P2P link`; GT=`<*> : Terminate AirDrop P2P link`

Nested coordinate pairs are collapsed into fewer slots in every label-batch generation. Similar failing coordinate reference and just-above-threshold risk support a structural association, not a precise confidence claim.


## C018: llama/standard/Hadoop/TP

ID `hadoop-23951`, index246, seed0, slotnear_threshold; seen GT=False. Risk83.315088, threshold82.451607, percentile45.00%, margin/IQR0.0060. First/last votes3/5,1/5; failure label=1. Other method percentiles: Clotho=45.0%, lexical_LR=58.2%, raw_hidden_LR=53.3%, same_space_LR=95.5%, same_space_LR_real_only=95.5%, same_space_two_density=51.1%, structural_LR=12.6%

Raw log:
```text
In stop, writing event MAP_ATTEMPT_FAILED
```
GT:
```text
In stop, writing event <*>
```
Sample0:
```text
In stop, writing event <*>
```

Target-batch parsed outputs:
- Sample5: `In <*> writing event MAP_ATTEMPT_FAILED`
- Sample6: `In stop, writing event <*>*`
- Sample7: `In stop, writing <*> MAP_ATTEMPT_FAILED`
- Sample8: `In stop, writing event MAP_ATTEMPT_FAILED`
- Sample9: `In stop, writing event <*>`

Nearest labeled references:
- Index131, FAIL=False: `Merging 8 sorted segments`; GT=`Merging <*> sorted segments`
- Index113, FAIL=True: `Notify RMCommunicator isAMLastRetry: true`; GT=`Notify RMCommunicator isAMLastRetry: <*>`
- Index305, FAIL=False: `Ramping up 1`; GT=`Ramping up <*>`

Sample0 is correct but label-batch 1/5 succeeds; masking event-name words varies. TP is a correct majority prediction with an unstable opposite first batch, illustrating the target differs from sample0.


## C019: llama/standard/Zookeeper/TP

ID `zookeeper-74270`, index127, seed0, slotseeded_representative; seen GT=True. Risk149.949945, threshold-6.667789, percentile85.73%, margin/IQR2.4081. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=85.7%, lexical_LR=92.7%, raw_hidden_LR=84.8%, same_space_LR=91.3%, same_space_LR_real_only=92.3%, same_space_two_density=85.7%, structural_LR=81.8%

Raw log:
```text
Follower sid: 2 : info : org.apache.zookeeper.server.quorum.QuorumPeer$QuorumServer@7761c32f
```
GT:
```text
Follower sid: <*> : info : <*>
```
Sample0:
```text
Follower sid: <*> : info : org.apache.zookeeper.server.quorum.QuorumPeer$QuorumServer@<>
```

Target-batch parsed outputs:
- Sample5: `Follower sid: <*> : info : org.apache.zookeeper.server.quorum.QuorumPeer$QuorumServer@<*>`
- Sample6: `Follower sid: <*> : info : org.apache.zookeeper.server.quorum.QuorumPeer$QuorumServer@<*>`
- Sample7: `Follower sid: 2 : info : org.apache.zookeeper.server.quorum.QuorumPeer$QuorumServer@<*>`
- Sample8: `Follower sid: 2 : info : org.apache.zookeeper.server.quorum.QuorumPeer$QuorumServer@<*>`
- Sample9: `Follower sid:  : info : org.apache.zookeeper.server.quorum.QuorumPeer$QuorumServer@`

Nearest labeled references:
- Index218, FAIL=True: `Follower sid: 1 : info : org.apache.zookeeper.server.quorum.QuorumPeer$QuorumServer@274c150e`; GT=`Follower sid: <*> : info : <*>`
- Index192, FAIL=True: `Follower sid: 1 : info : org.apache.zookeeper.server.quorum.QuorumPeer$QuorumServer@52450ebf`; GT=`Follower sid: <*> : info : <*>`
- Index285, FAIL=True: `Follower sid: 1 : info : org.apache.zookeeper.server.quorum.QuorumPeer$QuorumServer@188ce6aa`; GT=`Follower sid: <*> : info : <*>`

Object identity is repeatedly retained; all ten fail and same-family reference also fails. Here familiarity and high risk agree, countering an explanation that Clotho invariably misses word-like variables.


## C020: llama/standard/Zookeeper/TP

ID `zookeeper-19136`, index493, seed0, slotseeded_representative; seen GT=True. Risk47.789497, threshold-6.667789, percentile73.29%, margin/IQR0.8373. First/last votes2/5,2/5; failure label=1. Other method percentiles: Clotho=73.3%, lexical_LR=61.1%, raw_hidden_LR=17.4%, same_space_LR=59.1%, same_space_LR_real_only=76.2%, same_space_two_density=72.6%, structural_LR=52.3%

Raw log:
```text
Snapshotting: 0x100001564 to /var/lib/zookeeper/version-2/snapshot.100001564
```
GT:
```text
Snapshotting: <*> to <*>
```
Sample0:
```text
Snapshotting: <*> to <*>/<*>.<*>.
```

Target-batch parsed outputs:
- Sample5: `Snapshotting: <*> to <*>.`
- Sample6: `Snapshotting: <*> to <*>.`
- Sample7: `Snapshotting: <*> to <*>`
- Sample8: `Snapshotting: <*> to <*>.`
- Sample9: `Snapshotting: <*> to <*>`

Nearest labeled references:
- Index290, FAIL=True: `Snapshotting: 0xb00000084 to /var/lib/zookeeper/version-2/snapshot.b00000084`; GT=`Snapshotting: <*> to <*>`
- Index253, FAIL=False: `Reading snapshot /var/lib/zookeeper/version-2/snapshot.d0000001b`; GT=`Reading snapshot <*>`
- Index439, FAIL=True: `Expiring session 0x34ed9ac1c1e0040, timeout of 10000ms exceeded`; GT=`Expiring session <*> timeout of <*> exceeded`

Snapshot path segmentation and appended punctuation fluctuate around majority. Same-family failing neighbor and higher risk contrast C014's FP within the same broad family.


## C021: llama/standard/BGL/TP

ID `bgl-4529278`, index449, seed0, slotseeded_representative; seen GT=True. Risk1347.494546, threshold27.033638, percentile98.90%, margin/IQR14.3637. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=98.9%, lexical_LR=88.9%, raw_hidden_LR=99.9%, same_space_LR=99.9%, same_space_LR_real_only=99.9%, same_space_two_density=98.9%, structural_LR=99.4%

Raw log:
```text
dear 0x01fc5640 (543 RAS KERNEL INFO Kernel detected 5438714 integer alignment exceptions (5438707) iar 0x00544ea8, dear 0x01fc5620 (5438708) iar 0x00544eb8, dear 0x01fc5640 (5438709) iar 0x00544ea8, dear 0x01fc5660 (5438710) iar 0x00544eb8, dear 0x01fc5680 (5438711) iar 0x00544ee0, dear 0x01fc56a0 (5438712) iar 0x00544ef0, dear 0x01fc56c0 (5438713) iar 0x00544ee0, dear 0x01fc56e0 (5438714) iar 0x00544ef0, dear 0x01fc5700
```
GT:
```text
dear <*> (<*> RAS KERNEL INFO Kernel detected <*> integer alignment exceptions (<*>) iar <*>, dear <*> (<*>) iar <*>, dear <*> (<*>) iar <*>, dear <*> (<*>) iar <*>, dear <*> (<*>) iar <*>, dear <*> (<*>) iar <*>, dear <*> (<*>) iar <*>, dear <*> (<*>) iar <*>, dear <*>
```
Sample0:
```text
dear <*> (543 RAS KERNEL INFO Kernel detected 5438714 integer alignment exceptions (5438707) iar <*> dear <*> (5438708) iar <*> dear <*> (5438709) iar <*> dear <*> (5438710) iar <*> dear <*> (5438711) iar <*> dear <*> (5438712) iar <*> dear <*> (5438713) iar <*> dear <*> (5438714) iar <*>
```

Target-batch parsed outputs:
- Sample5: `dear <*> (543 RAS KERNEL INFO Kernel detected 5438714 integer alignment exceptions (5438707) iar <*> , dear <*> (5438708) iar <*> , dear <*> (5438709) iar <*> , dear <*> (5438710) iar <*> , dear <*> (5438711) iar <*> , dear <*> (5438712) iar <*> , dear <*> (5438713) iar <*> , dear <*> (5438714) iar <*> , dear <*>`
- Sample6: `dear <*> (<*>) KERNEL INFO Kernel detected <*> integer alignment exceptions (<*>) iar <*>`
- Sample7: `dear <*> (543 RAS KERNEL INFO Kernel detected 5438714 integer alignment exceptions (5438707) iar <*> , dear <*> (5438708) iar <*> , dear <*> (5438709) iar <*> , dear <*> (5438710) iar <*> , dear <*> (5438711) iar <*> , dear <*> (5438712) iar <*> , dear <*> (5438713) iar <*> , dear <*> (5438714) iar <*> , dear <*>`
- Sample8: `dear <*> (<* RAS KERNEL INFO <*> integer alignment exceptions (<*>) iar <*>>, dear <*> (<*>) iar <*>>, dear <*> (<*>) iar <*>>, dear <*> (<*>) iar <*>, dear <*> (<*>) iar <*>>, dear <*> (<*>) iar <*>, dear <*> (<*>) iar <*>>, dear <*> (<*>) iar <*>, dear <*> (<*>) iar <*>>, dear <*> (<*>) iar <*>>`
- Sample9: `dear <*> (543 RAS KERNEL INFO Kernel detected 5438714 integer alignment exceptions (5438707) iar <*> , dear <*> (5438708) iar <*> , dear <*> (5438709) iar <*> , dear <*> (5438710) iar <*> , dear <*> (5438711) iar <*> , dear <*> (5438712) iar <*> , dear <*> (5438713) iar <*> , dear <*> (5438714) iar <*>`

Nearest labeled references:
- Index69, FAIL=True: `dear 0x01fc1e20 (421 RAS KERNEL INFO Kernel detected 4211600 integer alignment exceptions (4211593) iar 0x00544ea8, dear 0x01fc1e00 (4211594) iar 0x00544eb8, dear 0x01fc1e20 (4211595) iar 0x00544ea8, dear 0x01fc1e40 (4211596) iar 0x00544eb8, dear 0x01fc1e60 (4211597) iar 0x00544ee0, dear 0x01fc1e80 (4211598) iar 0x00544ef0, dear 0x01fc1ea0 (4211599) iar 0x00544ee0, dear 0x01fc1ec0 (4211600) iar 0x00544ef0, dear 0x01fc1ee0`; GT=`dear <*> (<*> RAS KERNEL INFO Kernel detected <*> integer alignment exceptions (<*>) iar <*>, dear <*> (<*>) iar <*>, dear <*> (<*>) iar <*>, dear <*> (<*>) iar <*>, dear <*> (<*>) iar <*>, dear <*> (<*>) iar <*>, dear <*> (<*>) iar <*>, dear <*> (<*>) iar <*>, dear <*>`
- Index88, FAIL=True: `Kernel detected 4035396 integer alignment exceptions (4035389) iar 0x00544ea8, dear 0x01ef7e60 (4035390) iar 0x00544eb8, dear 0x01ef7e80 (4035391) iar 0x00544ea8, dear 0x01ef7ea0 (4035392) iar 0x00544eb8, dear 0x01ef7ec0 (4035393) iar 0x00544ee0, dear 0x01ef7ee0 (4035394) iar 0x00544ef0, dear 0x01ef7f00 (4035395) iar 0x00544ee0, dear 0x01ef7f20 (4035396) iar 0x00544ef0, dear 0x01ef7f40`; GT=`Kernel detected <*> integer alignment exceptions (<*>) iar <*>, dear <*> (<*>) iar <*>, dear <*> (<*>) iar <*>, dear <*> (<*>) iar <*>, dear <*> (<*>) iar <*>, dear <*> (<*>) iar <*>, dear <*> (<*>) iar <*>, dear <*> (<*>) iar <*>, dear <*>`
- Index27, FAIL=True: `Ido chip status changed: FF:F2:9F:16:E8:83:00:0D:60:E9:17:7C ip=10.0.1.7 v=13 t=4 status=M Tue Sep 20 09:28:22 PDT 2005`; GT=`Ido chip status changed: <*> ip=<*> v=<*> t=<*> status=<*> <*>`

Long repeated register message has many retained constants/variables and all ten fail. Same-family failure is correctly high-risk; structural length alone cannot explain low-risk failures C005/C006.


## C022: llama/standard/Zookeeper/TN

ID `zookeeper-44809`, index199, seed0, slotextreme; seen GT=True. Risk-105.418342, threshold-6.667789, percentile0.12%, margin/IQR-1.5184. First/last votes5/5,5/5; failure label=0. Other method percentiles: Clotho=0.1%, lexical_LR=0.5%, raw_hidden_LR=7.4%, same_space_LR=45.5%, same_space_LR_real_only=40.9%, same_space_two_density=54.3%, structural_LR=16.2%

Raw log:
```text
Received connection request /10.10.34.13:45946
```
GT:
```text
Received connection request <*>
```
Sample0:
```text
Received connection request <*>
```

Target-batch parsed outputs:
- Sample5: `Received connection request <*>`
- Sample6: `Received connection request <*>`
- Sample7: `Received connection request <*>`
- Sample8: `Received connection request <*>`
- Sample9: `Received connection request <*>`

Nearest labeled references:
- Index327, FAIL=False: `Received connection request /10.10.34.12:49455`; GT=`Received connection request <*>`
- Index422, FAIL=False: `Received connection request /10.10.34.11:58842`; GT=`Received connection request <*>`
- Index309, FAIL=False: `Received connection request /10.10.34.11:58805`; GT=`Received connection request <*>`

All ten successfully abstract a connection address; near-identical passing reference and very low risk are compatible with H1. Compare unfamiliar but successful C026.


## C023: llama/standard/Zookeeper/TN

ID `zookeeper-3373`, index133, seed0, slotextreme; seen GT=True. Risk-105.358785, threshold-6.667789, percentile0.37%, margin/IQR-1.5174. First/last votes5/5,4/5; failure label=0. Other method percentiles: Clotho=0.4%, lexical_LR=1.3%, raw_hidden_LR=5.0%, same_space_LR=39.6%, same_space_LR_real_only=33.3%, same_space_two_density=54.0%, structural_LR=16.2%

Raw log:
```text
Received connection request /10.10.34.11:46144
```
GT:
```text
Received connection request <*>
```
Sample0:
```text
Received connection request <*>
```

Target-batch parsed outputs:
- Sample5: `Received connection request /<*>/:<*>`
- Sample6: `Received connection request <*>`
- Sample7: `Received connection request <*>`
- Sample8: `Received connection request <*>`
- Sample9: `Received connection request <*>`

Nearest labeled references:
- Index422, FAIL=False: `Received connection request /10.10.34.11:58842`; GT=`Received connection request <*>`
- Index123, FAIL=False: `Received connection request /10.10.34.11:46101`; GT=`Received connection request <*>`
- Index309, FAIL=False: `Received connection request /10.10.34.11:58805`; GT=`Received connection request <*>`

Connection-request family is mostly correct, with one boundary-splitting failure. Low risk remains appropriate to the majority; a stochastic failure alone does not make it FN.


## C024: llama/standard/Apache/TN

ID `apache-16075`, index199, seed0, slotnear_threshold; seen GT=True. Risk-34.311573, threshold-34.308093, percentile68.41%, margin/IQR-0.0016. First/last votes3/5,4/5; failure label=0. Other method percentiles: Clotho=68.4%, lexical_LR=13.0%, raw_hidden_LR=37.2%, same_space_LR=51.1%, same_space_LR_real_only=50.6%, same_space_two_density=13.3%, structural_LR=77.1%

Raw log:
```text
jk2_init() Found child 15319 in scoreboard slot 1
```
GT:
```text
jk2_init() Found child <*> in scoreboard slot <*>
```
Sample0:
```text
jk2_init() Found child <*> in scoreboard slot 1
```

Target-batch parsed outputs:
- Sample5: `jk2_init() Found child <*> in scoreboard slot <*>`
- Sample6: `jk2_init() Found child <*> in scoreboard slot <*>`
- Sample7: `jk2_init() Found child <*> in scoreboard slot <*>`
- Sample8: `jk2_init() Found child <*> in scoreboard slot <*>`
- Sample9: `jk2_init() Found child <-> in scoreboard slot <->`

Nearest labeled references:
- Index129, FAIL=False: `jk2_init() Found child 3143 in scoreboard slot 4`; GT=`jk2_init() Found child <*> in scoreboard slot <*>`
- Index189, FAIL=False: `jk2_init() Found child 7906 in scoreboard slot 9`; GT=`jk2_init() Found child <*> in scoreboard slot <*>`
- Index29, FAIL=False: `jk2_init() Can't find child 25118 in scoreboard`; GT=`jk2_init() Can't find child <*> in scoreboard`

Sample0 retains the slot number but 4/5 label-batch outputs succeed. Near-threshold TN shows why inspecting only the first parser output would misclassify predictor behavior.


## C025: llama/standard/Spark/TN

ID `spark-205004`, index9, seed0, slotnear_threshold; seen GT=False. Risk73.704484, threshold74.455715, percentile47.68%, margin/IQR-0.0061. First/last votes2/5,3/5; failure label=0. Other method percentiles: Clotho=47.7%, lexical_LR=46.6%, raw_hidden_LR=60.6%, same_space_LR=46.7%, same_space_LR_real_only=40.6%, same_space_two_density=61.1%, structural_LR=10.9%

Raw log:
```text
Missing parents: List(ShuffleMapStage 259)
```
GT:
```text
Missing parents: List(<*>)
```
Sample0:
```text
Missing parents: List(ShuffleMapStage <*> )
```

Target-batch parsed outputs:
- Sample5: `Missing parents: List(<*>  <*>)`
- Sample6: `Missing parents: List(<*>)`
- Sample7: `Missing parents: List(<*>)`
- Sample8: `Missing parents: List(<*>)`
- Sample9: `Missing parents: List(<*>Stage <*>`

Nearest labeled references:
- Index168, FAIL=True: `Parents of final stage: List(ShuffleMapStage 98)`; GT=`Parents of final stage: List(ShuffleMapStage <*>)`
- Index323, FAIL=True: `waiting: Set(ResultStage 99)`; GT=`waiting: Set(ResultStage <*>)`
- Index34, FAIL=True: `Got told to re-register updating block broadcast_14_piece0`; GT=`Got told to re-register updating block <*>`

Two first-batch versus three last-batch successes reverse the label. Similar failing stage-list reference does not prevent just-below-threshold success; weak evidence for a stable mechanism.


## C026: llama/standard/Spark/TN

ID `spark-14372776`, index225, seed0, slotseeded_representative; seen GT=False. Risk12.334119, threshold74.455715, percentile15.98%, margin/IQR-0.5084. First/last votes5/5,5/5; failure label=0. Other method percentiles: Clotho=16.0%, lexical_LR=9.4%, raw_hidden_LR=12.8%, same_space_LR=8.4%, same_space_LR_real_only=7.7%, same_space_two_density=13.3%, structural_LR=28.7%

Raw log:
```text
Removing executor 1 with no recent heartbeats: 165602 ms exceeds timeout 120000 ms
```
GT:
```text
Removing executor <*> with no recent heartbeats: <*> ms exceeds timeout <*> ms
```
Sample0:
```text
Removing executor <*> with no recent heartbeats: <*> ms exceeds timeout <*> ms
```

Target-batch parsed outputs:
- Sample5: `Removing executor <*> with no recent heartbeats: <*> ms exceeds timeout <*> ms`
- Sample6: `Removing executor <*> with no recent heartbeats: <*> ms exceeds timeout <*> ms`
- Sample7: `Removing executor <*> with no recent heartbeats: <*> ms exceeds timeout <*> ms`
- Sample8: `Removing executor <*> with no recent heartbeats: <*> ms exceeds timeout <*> ms`
- Sample9: `Removing executor <*> with no recent heartbeats: <*> ms exceeds timeout <*> ms`

Nearest labeled references:
- Index333, FAIL=False: `Lost executor 9 on mesos-slave-06: Executor heartbeat timed out after 154834 ms`; GT=`Lost executor <*> on <*>: Executor heartbeat timed out after <*> ms`
- Index101, FAIL=False: `Lost executor 4 on mesos-slave-11: Executor heartbeat timed out after 147429 ms`; GT=`Lost executor <*> on <*>: Executor heartbeat timed out after <*> ms`
- Index245, FAIL=False: `Driver 10.10.34.32:58820 disassociated! Shutting down.`; GT=`Driver <*> disassociated! Shutting down.`

All ten mask executor and two timing values correctly on an unseen template. A passing heartbeat family supports generalization beyond exact identity; counters novelty-as-failure.


## C027: llama/standard/Linux/TN

ID `linux-91`, index363, seed0, slotseeded_representative; seen GT=False. Risk56.151746, threshold89.687182, percentile23.78%, margin/IQR-0.2837. First/last votes2/5,3/5; failure label=0. Other method percentiles: Clotho=23.8%, lexical_LR=11.6%, raw_hidden_LR=44.0%, same_space_LR=28.7%, same_space_LR_real_only=30.6%, same_space_two_density=41.6%, structural_LR=38.2%

Raw log:
```text
agpgart: AGP aperture is 64M @ 0xf4000000
```
GT:
```text
agpgart: AGP aperture is <*> @ <*>
```
Sample0:
```text
agpgart: AGP aperture is <*> @ <*>
```

Target-batch parsed outputs:
- Sample5: `agpgart: AGP aperture is <*> @ <*>`
- Sample6: `agpgart: AGP aperture is <*> @ <*>`
- Sample7: `agpgart: AGP aperture is <*> @ 0x<*>`
- Sample8: `agpgart: AGP aperture is <*> @ <*>`
- Sample9: `agpgart: AGP aperture is <*>M @ <*>`

Nearest labeled references:
- Index333, FAIL=True: `PCI: Found IRQ 11 for device 0000:00:1f.2`; GT=`PCI: Found IRQ <*> for device <*>`
- Index495, FAIL=False: `BIOS-e820: 0000000007eae000 - 0000000008000000 (reserved)`; GT=`BIOS-e820: <*> - <*> (reserved)`
- Index183, FAIL=True: `Mount-cache hash table entries: 512 (order: 0, 4096 bytes)`; GT=`Mount-cache hash table entries: <*> (order: <*>, <*> bytes)`

Size-unit and hex-prefix boundaries alternate, but the majority succeeds narrowly. A failing PCI neighbor does not prevent low risk; H3 is plausible for this threshold-sensitive case.


## C028: llama/standard/Apache/TN

ID `apache-45487`, index448, seed0, slotseeded_representative; seen GT=True. Risk-34.528603, threshold-34.308093, percentile63.05%, margin/IQR-0.0986. First/last votes3/5,4/5; failure label=0. Other method percentiles: Clotho=63.0%, lexical_LR=65.7%, raw_hidden_LR=86.2%, same_space_LR=62.1%, same_space_LR_real_only=50.1%, same_space_two_density=59.6%, structural_LR=33.5%

Raw log:
```text
[client 212.3.249.71] script not found or unable to stat: /var/www/cgi-bin/awstats
```
GT:
```text
[client <*>] script not found or unable to stat: <*>
```
Sample0:
```text
[client <*>] script not found or unable to stat: <*>
```

Target-batch parsed outputs:
- Sample5: `[client <*>] script not found or unable to stat: <*>`
- Sample6: `[client <*>] script not found or unable to stat: <*>`
- Sample7: `[client <*>] script not found or unable to stat: /var/www/cgi-bin/awstats`
- Sample8: `[client <*>] script not found or unable to stat: <*>`
- Sample9: `[client <*>] script not found or unable to stat: <*>`

Nearest labeled references:
- Index204, FAIL=False: `[client 69.64.38.143] script not found or unable to stat: /var/www/cgi-bin/awstats`; GT=`[client <*>] script not found or unable to stat: <*>`
- Index210, FAIL=True: `[client 202.133.98.6] script not found or unable to stat: /var/www/cgi-bin/stats`; GT=`[client <*>] script not found or unable to stat: <*>`
- Index319, FAIL=True: `[client 200.123.168.156] script not found or unable to stat: /var/www/cgi-bin/awstats.pl`; GT=`[client <*>] script not found or unable to stat: <*>`

Mostly correct path abstraction and passing same-family reference produce TN. This counters C008: path variables and Apache family membership alone do not determine prediction failure.


## C029: llama/disjoint/HPC/FN

ID `hpc-385286`, index129, seed0, slotextreme; seen GT=False. Risk12.978292, threshold199.452743, percentile0.21%, margin/IQR-1.4996. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=0.2%, U_published=66.6%, lexical_LR=75.8%, raw_hidden_LR=38.9%, same_space_LR=41.0%, same_space_LR_real_only=46.0%, same_space_two_density=25.0%, structural_LR=63.2%

Raw log:
```text
boot (command 3433) Error: Console-Busy Port already in use
```
GT:
```text
boot (command <*>) Error: Console-Busy Port already in use
```
Sample0:
```text
boot (<*>) Error: Console-Busy Port already in use
```

Target-batch parsed outputs:
- Sample5: `boot (<*>) Error: Console-Busy Port already in use`
- Sample6: `boot (command <*>) Error: <*> Port already in use`
- Sample7: `boot (<*)> Error: Console-Busy <*> already in use`
- Sample8: `boot (command <*> ) Error: <*>-Busy Port already in use`
- Sample9: `boot (<*>) Error: <*> Port already in use.`

Nearest labeled references:
- Index432, FAIL=True: `boot (command 2341) Error: node went to SRM while trying to boot`; GT=`boot (command <*>) Error: node went to SRM while trying to boot`
- Index35, FAIL=True: `boot (command 2836) Error: node went to SRM while trying to boot`; GT=`boot (command <*>) Error: node went to SRM while trying to boot`
- Index23, FAIL=True: `boot (command 2142) Error: node went to SRM while trying to boot`; GT=`boot (command <*>) Error: node went to SRM while trying to boot`

All ten fail command-word or Error-phrase boundaries, yet this unseen line has bottom-ranked risk and a failing boot neighbor. Strong counterexample to insufficient familiarity alone; compare C051's same-GT stochastic success.


## C030: llama/disjoint/HPC/FN

ID `hpc-68097`, index319, seed0, slotextreme; seen GT=False. Risk15.072027, threshold199.452743, percentile0.63%, margin/IQR-1.4828. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=0.6%, U_published=52.3%, lexical_LR=2.3%, raw_hidden_LR=10.7%, same_space_LR=20.0%, same_space_LR_real_only=20.4%, same_space_two_density=13.7%, structural_LR=6.7%

Raw log:
```text
ServerFileSystem: ServerFileSystem domain storage342 is full
```
GT:
```text
ServerFileSystem: ServerFileSystem domain <*> is full
```
Sample0:
```text
ServerFileSystem: ServerFileSystem <*> storage <*> is full
```

Target-batch parsed outputs:
- Sample5: `Server<*>: Server<*> domain <*> is full`
- Sample6: `ServerFileSystem: ServerFileSystem <*> is full`
- Sample7: `ServerFileSystem: ServerFileSystem <*> is full`
- Sample8: `ServerFileSystem: ServerFileSystem <*> is full.`
- Sample9: `ServerFileSystem: ServerFileSystem <*> is full`

Nearest labeled references:
- Index409, FAIL=False: `ServerFileSystem: An ServerFileSystem domain panic has occurred on storage442`; GT=`ServerFileSystem: An ServerFileSystem domain panic has occurred on <*>`
- Index150, FAIL=True: `ServerFileSystem: An ServerFileSystem domain panic has occurred on storage1223`; GT=`ServerFileSystem: An ServerFileSystem domain panic has occurred on <*>`
- Index49, FAIL=False: `ServerFileSystem: An ServerFileSystem domain panic has occurred on storage911`; GT=`ServerFileSystem: An ServerFileSystem domain panic has occurred on <*>`

All ten fail word-like storage-domain boundaries near a successful ServerFileSystem family. Low risk confuses a familiar prefix with correct abstraction; compare correct word-variable C056.


## C031: llama/disjoint/Thunderbird/FN

ID `thunderbird-9430332`, index126, seed0, slotnear_threshold; seen GT=False. Risk157.346089, threshold157.709783, percentile42.60%, margin/IQR-0.0022. First/last votes3/5,2/5; failure label=1. Other method percentiles: Clotho=42.6%, U_published=22.2%, lexical_LR=83.0%, raw_hidden_LR=33.0%, same_space_LR=10.6%, same_space_LR_real_only=13.4%, same_space_two_density=23.8%, structural_LR=27.4%

Raw log:
```text
Unknown username "haldaemon" in message bus configuration file
```
GT:
```text
Unknown username <*> in message bus configuration file
```
Sample0:
```text
Unknown username "<*>" in message bus configuration file
```

Target-batch parsed outputs:
- Sample5: `Unknown username <*> in message bus configuration file`
- Sample6: `Unknown username <*> in message bus configuration file`
- Sample7: `Unknown username "<*>" in message bus configuration file`
- Sample8: `Unknown username "<*>" in message bus configuration file`
- Sample9: `Unknown username "<*" in <*>`

Nearest labeled references:
- Index189, FAIL=True: `task migration cache decay timeout: 1 msecs.`; GT=`task migration cache decay timeout: <*> msecs.`
- Index188, FAIL=True: `process_limit: processing hard nofile 32768 for DEFAULT`; GT=`process_limit: processing hard nofile <*> for DEFAULT`
- Index94, FAIL=True: `configure: keyword "authenticate" unknown, line ignored`; GT=`configure: keyword <*> unknown, line ignored`

Quotes around username are inconsistently retained; first batch succeeds 3/5 and last 2/5. Near-threshold FN can arise from batch instability rather than a robust hidden-state miss.


## C032: llama/disjoint/Spark/FN

ID `spark-11762891`, index159, seed0, slotnear_threshold; seen GT=False. Risk88.745679, threshold89.276598, percentile31.93%, margin/IQR-0.0024. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=31.9%, U_published=55.6%, lexical_LR=11.4%, raw_hidden_LR=79.3%, same_space_LR=76.9%, same_space_LR_real_only=78.1%, same_space_two_density=33.1%, structural_LR=36.7%

Raw log:
```text
Successfully started service 'org.apache.spark.network.netty.NettyBlockTransferService' on port 36293.
```
GT:
```text
Successfully started service <*> on port <*>.
```
Sample0:
```text
Successfully started service '<*>' on port <*>.
```

Target-batch parsed outputs:
- Sample5: `Successfully started service 'org.apache.spark.network.netty.NettyBlockTransferService' on port <*>.`
- Sample6: `Successfully started service '<*>' on port<*>.`
- Sample7: `Successfully started service '<*>' on port <*>.`
- Sample8: `Successfully started service '<*>' on port <*>.`
- Sample9: `Successfully started service '<*> on port <*>'.`

Nearest labeled references:
- Index467, FAIL=True: `Stopped Spark web UI at http://10.10.34.24:42843`; GT=`Stopped Spark web UI at <*>`
- Index338, FAIL=True: `Removing block manager BlockManagerId(17, mesos-slave-27, 40642)`; GT=`Removing block manager BlockManagerId(<*>, <*>, <*>)`
- Index449, FAIL=True: `Stopped Spark web UI at http://10.10.34.20:55898`; GT=`Stopped Spark web UI at <*>`

All ten mishandle quotes or service-name scope. A failing web-UI reference and low-ish risk show a stable unseen-boundary miss, not just stochastic majority reversal.


## C033: llama/disjoint/HDFS/FN

ID `hdfs-5752508`, index6, seed0, slotseeded_representative; seen GT=False. Risk535.458759, threshold608.197646, percentile42.48%, margin/IQR-0.1499. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=42.5%, U_published=70.5%, lexical_LR=69.3%, raw_hidden_LR=47.8%, same_space_LR=59.1%, same_space_LR_real_only=64.0%, same_space_two_density=31.1%, structural_LR=26.0%

Raw log:
```text
PacketResponder blk_-539148231797001003 2 Exception java.io.EOFException
```
GT:
```text
PacketResponder <*> <*> Exception java.io.EOFException
```
Sample0:
```text
PacketResponder <*>, 2 Exception <*>.
```

Target-batch parsed outputs:
- Sample5: `PacketResponder <*> 2 Exception java.io.EOFException`
- Sample6: `PacketResponder <*> 2 Exception java.io.<*>Exception`
- Sample7: `PacketResponder <*> 2 Exception <*>`
- Sample8: `PacketResponder <*> 2 Exception java.io.EOFException`
- Sample9: `PacketResponder <*> 2 Exception <*>`

Nearest labeled references:
- Index485, FAIL=True: `Exception in receiveBlock for block blk_377785190538046925 java.io.EOFException`; GT=`Exception in receiveBlock for block <*> java.io.EOFException`
- Index85, FAIL=False: `Exception in receiveBlock for block blk_-6059290274455367169 java.io.EOFException`; GT=`Exception in receiveBlock for block <*> java.io.EOFException`
- Index16, FAIL=True: `PacketResponder blk_-6059290274455367169 2 Exception java.io.IOException: Connection reset by peer`; GT=`PacketResponder <*> <*> Exception java.io.IOException: Connection reset by peer`

All ten fail to jointly mask responder index and block ID or retain exception constants. A failing receiveBlock neighbor exists but score is below threshold; local failure similarity is insufficient.


## C034: llama/disjoint/Hadoop/FN

ID `hadoop-2689`, index451, seed0, slotseeded_representative; seen GT=False. Risk77.570960, threshold195.619264, percentile31.40%, margin/IQR-0.7043. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=31.4%, U_published=60.2%, lexical_LR=50.6%, raw_hidden_LR=47.4%, same_space_LR=35.4%, same_space_LR_real_only=31.8%, same_space_two_density=50.6%, structural_LR=53.4%

Raw log:
```text
TaskHeartbeatHandler thread interrupted
```
GT:
```text
TaskHeartbeatHandler thread interrupted
```
Sample0:
```text
Task<*>Handler thread <*>
```

Target-batch parsed outputs:
- Sample5: `Task<*>Handler <*>`
- Sample6: `Task<*> Handler <*>`
- Sample7: `Task<*>Handler thread interrupted`
- Sample8: `Task<*>Handler thread <*>.`
- Sample9: `Task<*>Handler thread interrupted`

Nearest labeled references:
- Index399, FAIL=True: `Process Thread Dump: Communication exception`; GT=`Process Thread Dump: Communication exception`
- Index288, FAIL=False: `In stop, writing event JOB_FINISHED`; GT=`In stop, writing event <*>`
- Index305, FAIL=False: `Ramping up 1`; GT=`Ramping up <*>`

A variable-free thread message is overmasked in every generation. Low risk near another failing thread message counters the claim that only conspicuous variable types create FN.


## C035: llama/disjoint/Hadoop/FN

ID `hadoop-55`, index158, seed0, slotseeded_representative; seen GT=False. Risk113.798842, threshold195.619264, percentile44.20%, margin/IQR-0.4882. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=44.2%, U_published=77.4%, lexical_LR=95.4%, raw_hidden_LR=90.2%, same_space_LR=96.6%, same_space_LR_real_only=98.2%, same_space_two_density=57.0%, structural_LR=94.8%

Raw log:
```text
Successfully connected to /10.190.173.170:50010 for BP-1347369012-10.190.173.170-1444972147527:blk_1073742826_2022
```
GT:
```text
Successfully connected to <*> for <*>
```
Sample0:
```text
Successfully connected to <*> for BP- <*>- <*>:blk_ <*>_<>
```

Target-batch parsed outputs:
- Sample5: `Successfully connected to <*> for <*>:blk_<*>`
- Sample6: `Successfully connected to <*> for <*>:blk_<>_<>`
- Sample7: `Successfully connected to <*> for <*>:blk_<>_<>`
- Sample8: `Successfully connected to <*>:50010 for <*>-<*>:*blk_*_*`
- Sample9: `Successfully connected to <*> for BP-<*>:<*>:blk_<*>_<*>`

Nearest labeled references:
- Index350, FAIL=True: `TaskAttempt killed because it ran on unusable node 04DN8IQ.fareast.corp.microsoft.com:55452. AttemptId:attempt_1445087491445_0001_m_000011_0`; GT=`TaskAttempt killed because it ran on unusable node <*>. AttemptId:<*>`
- Index436, FAIL=True: `DFSOutputStream ResponseProcessor exception for block BP-1347369012-10.190.173.170-1444972147527:blk_1073743034_2246`; GT=`DFSOutputStream ResponseProcessor exception for block <*>`
- Index13, FAIL=True: `Retrying connect to server: MININT-FNANLI5.fareast.corp.microsoft.com/10.86.169.121:55135. Already tried 0 time(s); maxRetries=45`; GT=`Retrying connect to server: <*>. Already tried <*> time(s); maxRetries=<*>`

Compound block-pool ID is fragmented in all ten. Nearby failed task message is loosely related; low risk shows familiar Hadoop vocabulary does not guarantee failure-sensitive geometry.


## C036: llama/disjoint/Thunderbird/FP

ID `thunderbird-1203`, index414, seed0, slotextreme; seen GT=False. Risk5441.387781, threshold157.709783, percentile98.20%, margin/IQR32.0529. First/last votes4/5,3/5; failure label=0. Other method percentiles: Clotho=98.2%, U_published=90.6%, lexical_LR=1.8%, raw_hidden_LR=77.8%, same_space_LR=80.2%, same_space_LR_real_only=91.8%, same_space_two_density=98.2%, structural_LR=41.0%

Raw log:
```text
Stopping syslog-ng:
```
GT:
```text
Stopping syslog-ng:
```
Sample0:
```text
Stopping syslog-ng: <>
```

Target-batch parsed outputs:
- Sample5: `Stopping syslog-ng:`
- Sample6: `Stopping syslog-ng:`
- Sample7: `Stopping syslog-ng: req_8812_03.`
- Sample8: `Stopping syslog-ng:`
- Sample9: `Stopping syslog-ng: <*>.`

Nearest labeled references:
- Index142, FAIL=True: `ACPI wakeup devices:`; GT=`ACPI wakeup devices:`
- Index101, FAIL=True: `** Ignoring requests on eth1:0. If this is not what`; GT=`** Ignoring requests on <*>. If this is not what`
- Index435, FAIL=True: `Support MySQL by buying support/licenses at https://#4#`; GT=`Support <*> by buying support/licenses at <*>`

Short service header has invented completions in some outputs, but both batches majority succeed. Very high risk overgeneralizes the same short-header behavior correctly flagged in C043/C044.


## C037: llama/disjoint/BGL/FP

ID `bgl-3445357`, index247, seed0, slotextreme; seen GT=False. Risk131.537696, threshold-46.292348, percentile97.00%, margin/IQR11.5995. First/last votes4/5,5/5; failure label=0. Other method percentiles: Clotho=97.0%, U_published=0.2%, lexical_LR=21.4%, raw_hidden_LR=3.8%, same_space_LR=0.2%, same_space_LR_real_only=0.2%, same_space_two_density=0.2%, structural_LR=0.2%

Raw log:
```text
program
```
GT:
```text
program
```
Sample0:
```text
program
```

Target-batch parsed outputs:
- Sample5: `program`
- Sample6: `program`
- Sample7: `program`
- Sample8: `program`
- Sample9: `program`

Nearest labeled references:
- Index149, FAIL=False: `Microloader Assertion`; GT=`Microloader Assertion`
- Index138, FAIL=False: `kernel panic`; GT=`kernel panic`
- Index89, FAIL=False: `shutdown complete`; GT=`shutdown complete`

One-word program is copied correctly in all ten despite extremely high risk. Successful nearest short assertion refutes a universal low-familiarity-implies-parser-failure account; compare C043.


## C038: llama/disjoint/OpenStack/FP

ID `openstack-143203`, index359, seed0, slotnear_threshold; seen GT=False. Risk0.901375, threshold0.883449, percentile48.38%, margin/IQR0.0013. First/last votes5/5,4/5; failure label=0. Other method percentiles: Clotho=48.4%, U_published=34.2%, lexical_LR=5.5%, raw_hidden_LR=19.6%, same_space_LR=15.6%, same_space_LR_real_only=8.7%, same_space_two_density=51.2%, structural_LR=3.2%

Raw log:
```text
[instance: 32235e06-d497-473d-8a32-3cbe1a6a7e3b] VM Resumed (Lifecycle Event)
```
GT:
```text
[instance: <*>] VM Resumed (Lifecycle Event)
```
Sample0:
```text
[instance: <*>] VM Resumed (Lifecycle Event)
```

Target-batch parsed outputs:
- Sample5: `[instance: <*>] VM Resumed (Lifecycle Event)`
- Sample6: `[instance: <*>] VM Resumed (Lifecycle Event)`
- Sample7: `[instance: <*>] VM Resumed (Lifecycle Event)`
- Sample8: `[instance: <*>] <*> Resumed (Lifecycle Event)`
- Sample9: `[instance: <*>] VM Resumed (Lifecycle Event)`

Nearest labeled references:
- Index12, FAIL=False: `[instance: dd06a15b-3d81-47b9-9ec1-273fd3e4dce2] VM Paused (Lifecycle Event)`; GT=`[instance: <*>] VM Paused (Lifecycle Event)`
- Index241, FAIL=False: `[instance: 6c27f39a-940d-43f0-a71d-c73d8f867bb4] VM Stopped (Lifecycle Event)`; GT=`[instance: <*>] VM Stopped (Lifecycle Event)`
- Index491, FAIL=False: `[instance: 1ca3f50d-5335-4bf1-969c-5b0f7fd110d6] VM Started (Lifecycle Event)`; GT=`[instance: <*>] VM Started (Lifecycle Event)`

Resumed lifecycle template is unseen but close to a successful Paused family and mostly parsed correctly. Near-threshold FP is weak evidence; compare low-risk lifecycle success C110.


## C039: llama/disjoint/Linux/FP

ID `linux-64`, index180, seed0, slotnear_threshold; seen GT=False. Risk-40.935648, threshold-40.960396, percentile25.40%, margin/IQR0.0016. First/last votes2/5,3/5; failure label=0. Other method percentiles: Clotho=25.4%, U_published=89.0%, lexical_LR=67.4%, raw_hidden_LR=15.4%, same_space_LR=65.8%, same_space_LR_real_only=61.4%, same_space_two_density=15.8%, structural_LR=34.6%

Raw log:
```text
PCI: Invalid ACPI-PCI IRQ routing table
```
GT:
```text
PCI: Invalid ACPI-PCI IRQ routing table
```
Sample0:
```text
PCI: Invalid <*>-<*>
```

Target-batch parsed outputs:
- Sample5: `PCI: Invalid ACPI-PCI IRQ routing table`
- Sample6: `PCI: Invalid <*>-PCI <*> routing table`
- Sample7: `PCI: Invalid ACPI-<*> IRQ routing table`
- Sample8: `PCI: Invalid ACPI-PCI IRQ routing table`
- Sample9: `PCI: Invalid ACPI-PCI IRQ routing table`

Nearest labeled references:
- Index81, FAIL=True: `isapnp: No Plug & Play device found`; GT=`isapnp: No Plug & Play device found`
- Index235, FAIL=True: `EXT3-fs: mounted filesystem with ordered data mode.`; GT=`EXT3-fs: mounted filesystem with ordered data mode.`
- Index2, FAIL=True: `Bluetooth: HCI socket layer initialized`; GT=`Bluetooth: HCI socket layer initialized`

Variable-free PCI phrase is sometimes overmasked and changes majority between batches. Near-threshold FP is compatible with H3; compare stable correct no-variable C093.


## C040: llama/disjoint/Spark/FP

ID `spark-2020473`, index437, seed0, slotseeded_representative; seen GT=False. Risk145.134587, threshold89.276598, percentile50.40%, margin/IQR0.2548. First/last votes5/5,5/5; failure label=0. Other method percentiles: Clotho=50.4%, U_published=3.8%, lexical_LR=59.2%, raw_hidden_LR=7.0%, same_space_LR=1.4%, same_space_LR_real_only=1.8%, same_space_two_density=13.9%, structural_LR=21.3%

Raw log:
```text
Disabling executor 16.
```
GT:
```text
Disabling executor <*>.
```
Sample0:
```text
Disabling executor <*>.
```

Target-batch parsed outputs:
- Sample5: `Disabling executor <*>.`
- Sample6: `Disabling executor <*>.`
- Sample7: `Disabling executor <*>.`
- Sample8: `Disabling executor <*>.`
- Sample9: `Disabling executor <*>.`

Nearest labeled references:
- Index253, FAIL=False: `Got assigned task 151786`; GT=`Got assigned task <*>`
- Index412, FAIL=False: `Cleaned accumulator 115`; GT=`Cleaned accumulator <*>`
- Index260, FAIL=True: `Registered signal handlers for [TERM, HUP, INT]`; GT=`Registered signal handlers for <*>`

All ten correctly mask executor index despite unseen template and modestly high score. A successful numeric-task neighbor is available; high density novelty is not identical to failure risk.


## C041: llama/disjoint/Spark/FP

ID `spark-272`, index478, seed0, slotseeded_representative; seen GT=False. Risk112.600782, threshold89.276598, percentile39.96%, margin/IQR0.1064. First/last votes5/5,5/5; failure label=0. Other method percentiles: Clotho=40.0%, U_published=1.0%, lexical_LR=5.4%, raw_hidden_LR=0.2%, same_space_LR=15.9%, same_space_LR_real_only=10.6%, same_space_two_density=50.0%, structural_LR=26.3%

Raw log:
```text
Remoting shut down.
```
GT:
```text
Remoting shut down.
```
Sample0:
```text
Remoting shut down.
```

Target-batch parsed outputs:
- Sample5: `Remoting shut down.`
- Sample6: `Remoting shut down.`
- Sample7: `Remoting shut down.`
- Sample8: `Remoting shut down.`
- Sample9: `Remoting shut down.`

Nearest labeled references:
- Index42, FAIL=True: `Stage 3 was cancelled`; GT=`Stage <*> was cancelled`
- Index234, FAIL=False: `Successfully stopped SparkContext`; GT=`Successfully stopped SparkContext`
- Index312, FAIL=False: `BlockManager stopped`; GT=`BlockManager stopped`

All ten copy the shutdown statement correctly. A failing stage-status neighbor accompanies FP, showing coarse semantic status similarity can transfer the wrong correctness pattern.


## C042: llama/disjoint/HDFS/FP

ID `hdfs-2457266`, index367, seed0, slotseeded_representative; seen GT=False. Risk685.719633, threshold608.197646, percentile62.40%, margin/IQR0.1598. First/last votes4/5,3/5; failure label=0. Other method percentiles: Clotho=62.4%, U_published=16.1%, lexical_LR=29.5%, raw_hidden_LR=30.7%, same_space_LR=7.5%, same_space_LR_real_only=18.5%, same_space_two_density=29.9%, structural_LR=69.1%

Raw log:
```text
Deleting block blk_2558137190651013904 file /mnt/hadoop/dfs/data/current/subdir46/blk_2558137190651013904
```
GT:
```text
Deleting block <*> file <*>
```
Sample0:
```text
Deleting block <*> file <*>
```

Target-batch parsed outputs:
- Sample5: `Deleting block <*> file /mnt/hadoop/dfs/data/current/subdir46/<*>`
- Sample6: `Deleting block <*> file <*>`
- Sample7: `Deleting block <*> file <*>`
- Sample8: `Deleting block <*> file /mnt/hadoop/dfs/data/current/subdir46/<*>`
- Sample9: `Deleting block <*> file <*>`

Nearest labeled references:
- Index466, FAIL=True: `10.251.30.6:50010:Transmitted block blk_-4262646822082772935 to /10.251.194.245:50010`; GT=`<*>:Transmitted block <*> to <*>`
- Index414, FAIL=True: `10.251.42.16:50010:Transmitted block blk_-1954047130407029757 to /10.250.18.114:50010`; GT=`<*>:Transmitted block <*> to <*>`
- Index256, FAIL=True: `10.251.37.240:50010:Transmitted block blk_1375833768007236572 to /10.251.107.98:50010`; GT=`<*>:Transmitted block <*> to <*>`

Path retention sometimes fails, but 3/5 label-batch generations succeed. High risk near a failing block-transfer reference is not evidence that all path abstractions fail.


## C043: llama/disjoint/Thunderbird/TP

ID `thunderbird-99676`, index110, seed0, slotextreme; seen GT=False. Risk9845.493348, threshold157.709783, percentile99.80%, margin/IQR58.7701. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=99.8%, U_published=85.0%, lexical_LR=78.6%, raw_hidden_LR=96.2%, same_space_LR=63.8%, same_space_LR_real_only=66.6%, same_space_two_density=99.8%, structural_LR=75.8%

Raw log:
```text
Mem-info:
```
GT:
```text
Mem-info:
```
Sample0:
```text
Mem-info:<*>.
```

Target-batch parsed outputs:
- Sample5: `None`
- Sample6: `None`
- Sample7: `None`
- Sample8: `None`
- Sample9: `Mem-info: from <IPaddress>:<port>`

Nearest labeled references:
- Index142, FAIL=True: `ACPI wakeup devices:`; GT=`ACPI wakeup devices:`
- Index101, FAIL=True: `** Ignoring requests on eth1:0. If this is not what`; GT=`** Ignoring requests on <*>. If this is not what`
- Index74, FAIL=True: `is enough space.`; GT=`is enough space.`

Variable-free Mem-info header produces extra fields or unparseable outputs in all ten and maximal risk. Compare C037/C093: superficially similar short headers can have opposite correctness.


## C044: llama/disjoint/Linux/TP

ID `linux-5371`, index319, seed0, slotextreme; seen GT=False. Risk815.214896, threshold-40.960396, percentile98.20%, margin/IQR54.0050. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=98.2%, U_published=38.2%, lexical_LR=56.6%, raw_hidden_LR=66.2%, same_space_LR=32.2%, same_space_LR_real_only=28.6%, same_space_two_density=98.2%, structural_LR=5.4%

Raw log:
```text
Stopping squid:
```
GT:
```text
Stopping squid:
```
Sample0:
```text
Stopping squid: <*> is going down after <*> is terminated.
```

Target-batch parsed outputs:
- Sample5: `Stopping <*>:.`
- Sample6: `None`
- Sample7: `Jul 18 05:08:10  <*> squid[<*>]: squid[<*>] process received <*> signal.`
- Sample8: `Stopping squid`
- Sample9: `Stopping squid: <*> has crashed on <*> with error: <*> failed to connect to <*> on '*' in <*>s: <*>`

Nearest labeled references:
- Index35, FAIL=True: `Starting background readahead:`; GT=`Starting background readahead:`
- Index291, FAIL=True: `*** info [mice.c(1766)]:`; GT=`*** info [mice.c(<*>)]:`
- Index10, FAIL=True: `BIOS-provided physical RAM map:`; GT=`BIOS-provided physical RAM map:`

Same raw line as C015 remains a TP under disjoint references, though score scale changes greatly. This is a counterexample to assuming every failure becomes missed after template removal.


## C045: llama/disjoint/OpenSSH/TP

ID `openssh-235825`, index211, seed0, slotnear_threshold; seen GT=False. Risk2644.382189, threshold2643.198028, percentile86.14%, margin/IQR0.0009. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=86.1%, U_published=81.7%, lexical_LR=99.0%, raw_hidden_LR=89.8%, same_space_LR=32.7%, same_space_LR_real_only=22.7%, same_space_two_density=97.0%, structural_LR=93.2%

Raw log:
```text
pam_unix(sshd:auth): authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost=191.201.195.56 user=root
```
GT:
```text
pam_unix(sshd:auth): authentication failure; logname=<*> uid=<*> euid=<*> tty=<*> ruser=<*> rhost=<*> user=<*>
```
Sample0:
```text
pam_unix(sshd:auth): authentication failure; logname= <*> uid=<> euid=<> tty=ssh ruser= rhost=<*> user=<*>
```

Target-batch parsed outputs:
- Sample5: `pam_unix(<*>:auth): authentication failure; logname=<*> uid=<*> euid=<*> tty=<*> ruser=<*> rhost=<*> user=<*>`
- Sample6: `pam_unix(sshd:auth): authentication failure; logname=<*> uid=<> euid=<*> tty=<*> ruser=<*> rhost=<*> user=<*>`
- Sample7: `pam_unix(sshd:auth): authentication failure; logname= uid=<*> euid=<*> tty=ssh ruser= rhost=<*> user=<*>`
- Sample8: `pam_unix(sshd:auth): authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost=<*> user=<*>`
- Sample9: `pam_unix(sshd:auth): authentication failure; logname= <*>  uid=<> euid=<> tty=<> ruser= <*> rhost=<*> user=<*>`

Nearest labeled references:
- Index165, FAIL=True: `PAM 4 more authentication failures; logname= uid=0 euid=0 tty=ssh ruser= rhost=192.151.237.154 user=root`; GT=`PAM <*> more authentication failures; logname=<*> uid=<*> euid=<*> tty=<*> ruser=<*> rhost=<*> user=<*>`
- Index55, FAIL=True: `PAM 5 more authentication failures; logname= uid=0 euid=0 tty=ssh ruser= rhost=1-36-177-127.static.netvigator.com user=root`; GT=`PAM <*> more authentication failures; logname=<*> uid=<*> euid=<*> tty=<*> ruser=<*> rhost=<*> user=<*>`
- Index356, FAIL=True: `PAM 5 more authentication failures; logname= uid=0 euid=0 tty=ssh ruser= rhost=1-164-122-177.dynamic-ip.hinet.net user=root`; GT=`PAM <*> more authentication failures; logname=<*> uid=<*> euid=<*> tty=<*> ruser=<*> rhost=<*> user=<*>`

Empty fields and punctuation in authentication record repeatedly fail. High risk near a failing authentication reference is consistent across-family generalization despite no identical GT template.


## C046: llama/disjoint/Linux/TP

ID `linux-142`, index179, seed0, slotnear_threshold; seen GT=False. Risk-40.923006, threshold-40.960396, percentile25.80%, margin/IQR0.0024. First/last votes0/5,1/5; failure label=1. Other method percentiles: Clotho=25.8%, U_published=26.6%, lexical_LR=93.8%, raw_hidden_LR=20.2%, same_space_LR=45.4%, same_space_LR_real_only=40.6%, same_space_two_density=14.2%, structural_LR=45.8%

Raw log:
```text
VFS: Mounted root (ext2 filesystem).
```
GT:
```text
VFS: Mounted root (ext2 filesystem).
```
Sample0:
```text
VFS: Mounted root (<*>).
```

Target-batch parsed outputs:
- Sample5: `VFS: Mounted <*> ( <*> filesystem).`
- Sample6: `VFS: Mounted <*> ( <*> filesystem).`
- Sample7: `VFS: Mounted <*> (<*> filesystem).`
- Sample8: `VFS: Mounted root (ext2 filesystem).`
- Sample9: `VFS: Mounted <*> ( <*> filesystem).`

Nearest labeled references:
- Index235, FAIL=True: `EXT3-fs: mounted filesystem with ordered data mode.`; GT=`EXT3-fs: mounted filesystem with ordered data mode.`
- Index81, FAIL=True: `isapnp: No Plug & Play device found`; GT=`isapnp: No Plug & Play device found`
- Index22, FAIL=False: `pci_hotplug: PCI Hot Plug PCI Core version: 0.5`; GT=`pci_hotplug: PCI Hot Plug PCI Core version: <*>`

A fixed filesystem phrase is overmasked in most generations. Threshold is barely crossed despite low population rank; outcome names should not be read as absolute high confidence.


## C047: llama/disjoint/Spark/TP

ID `spark-8438261`, index332, seed0, slotseeded_representative; seen GT=False. Risk274.257262, threshold89.276598, percentile75.30%, margin/IQR0.8436. First/last votes1/5,0/5; failure label=1. Other method percentiles: Clotho=75.3%, U_published=66.9%, lexical_LR=84.9%, raw_hidden_LR=68.1%, same_space_LR=23.1%, same_space_LR_real_only=13.9%, same_space_two_density=72.1%, structural_LR=63.1%

Raw log:
```text
Finished task 30.0 in stage 8977.0 (TID 359153). 2364 bytes result sent to driver
```
GT:
```text
Finished task <*> in stage <*> (TID <*>). <*> bytes result sent to driver
```
Sample0:
```text
Finished task <*> in stage <*> (TID <*>). <*> bytes result sent to <*>
```

Target-batch parsed outputs:
- Sample5: `Finished task <*>.0 in stage <*>.0 (TID <*>). <*>* bytes result sent to driver`
- Sample6: `Finished task <*> in stage <*>. (TID <*>). <*> bytes result sent to driver`
- Sample7: `Finished task <*>.0 in stage <*>.0 (TID <*>). <*>* bytes result sent to driver`
- Sample8: `Finished task <*> in stage <*>. (TID <*>). <*> bytes result sent to driver`
- Sample9: `Finished task <*>.0 in stage <*>.0 (TID <*>). <*> bytes result sent to driver`

Nearest labeled references:
- Index258, FAIL=True: `Lost task 1.0 in stage 3.0 (TID 9, mesos-master-2): java.lang.OutOfMemoryError: Requested array size exceeds VM limit`; GT=`Lost task <*> in stage <*> (TID <*>, <*>): java.lang.OutOfMemoryError: Requested array size exceeds VM limit`
- Index263, FAIL=True: `Container marked as failed: container_1485248649253_0124_02_000016 on host: mesos-slave-16. Exit status: -100. Diagnostics: Container expired since it was unused`; GT=`Container marked as failed: <*> on host: <*>. Exit status: <*>. Diagnostics: Container expired since it was unused`
- Index154, FAIL=True: `Error sending result RpcResponse{requestId=7635890109338441966, body=NioManagedBuffer{buf=java.nio.HeapByteBuffer[pos=0 lim=81 cap=81]}} to mesos-slave-25/10.10.34.35:34978; closing connection`; GT=`Error sending result RpcResponse{requestId=<*>, body=<*>} to <*>; closing connection`

Decimal task/stage suffixes are retained or punctuation changes in every label-batch output. Failing task-family neighbor and higher risk fit transfer of a structural failure family.


## C048: llama/disjoint/Thunderbird/TP

ID `thunderbird-1061`, index293, seed0, slotseeded_representative; seen GT=False. Risk497.934836, threshold157.709783, percentile89.80%, margin/IQR2.0639. First/last votes0/5,1/5; failure label=1. Other method percentiles: Clotho=89.8%, U_published=13.4%, lexical_LR=34.2%, raw_hidden_LR=29.4%, same_space_LR=79.0%, same_space_LR_real_only=65.8%, same_space_two_density=90.6%, structural_LR=85.8%

Raw log:
```text
Home page is http://#20#/
```
GT:
```text
Home page is <*>
```
Sample0:
```text
Home page is http://<*>
```

Target-batch parsed outputs:
- Sample5: `Home page is http://#/`
- Sample6: `Home page is http://<*>/`
- Sample7: `Home page is <*>`
- Sample8: `Home page is <*/*>.`
- Sample9: `Home page is http://#/`

Nearest labeled references:
- Index277, FAIL=True: `Sending warning via mail to #7#@localhost.localdomain ...`; GT=`Sending warning via mail to <*> ...`
- Index254, FAIL=True: `WARNING: DNS lookup failed for "#82#".`; GT=`WARNING: DNS lookup failed for "<*>".`
- Index189, FAIL=True: `task migration cache decay timeout: 1 msecs.`; GT=`task migration cache decay timeout: <*> msecs.`

URL scheme and separators are retained inconsistently. High risk correctly flags majority failure, but nearby mail-warning reference is not a direct family explanation.


## C049: llama/disjoint/Zookeeper/TP

ID `zookeeper-49535`, index166, seed0, slotseeded_representative; seen GT=False. Risk-32.163982, threshold-36.337216, percentile53.83%, margin/IQR0.1355. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=53.8%, U_published=52.1%, lexical_LR=87.9%, raw_hidden_LR=70.4%, same_space_LR=76.0%, same_space_LR_real_only=86.6%, same_space_two_density=70.4%, structural_LR=70.6%

Raw log:
```text
New election. My id = 2, proposed zxid=0x50000062e
```
GT:
```text
New election. My id = <*>, proposed <*>
```
Sample0:
```text
New election. My <*> = <*>, proposed zxid=<*>.
```

Target-batch parsed outputs:
- Sample5: `New election. My id = <*>, proposed zxid=<*>`
- Sample6: `New election. My id = <*>, proposed zxid=<*>.`
- Sample7: `New election. My id = <*>, proposed zxid=0x<*>.`
- Sample8: `New election. My <*> =<*> , proposed <*>=<*>`
- Sample9: `New election. My <*> = 2, proposed zxid=<*>.`

Nearest labeled references:
- Index151, FAIL=True: `Expiring session 0x34ed9ac1c1e00bb, timeout of 10000ms exceeded`; GT=`Expiring session <*> timeout of <*> exceeded`
- Index42, FAIL=True: `Expiring session 0x34ede65503f0043, timeout of 10000ms exceeded`; GT=`Expiring session <*> timeout of <*> exceeded`
- Index380, FAIL=True: `Got zxid 0xa00000001 expected 0x1`; GT=`Got zxid <*> expected <*>`

GT masks an entire proposed field rather than merely its value; every generation keeps key/punctuation. Risk correctly flags this convention-dependent failure without same-template references.


## C050: llama/disjoint/HPC/TN

ID `hpc-61348`, index92, seed0, slotextreme; seen GT=False. Risk21.964370, threshold199.452743, percentile7.35%, margin/IQR-1.4273. First/last votes4/5,4/5; failure label=0. Other method percentiles: Clotho=7.4%, U_published=2.7%, lexical_LR=24.6%, raw_hidden_LR=0.2%, same_space_LR=19.1%, same_space_LR_real_only=9.5%, same_space_two_density=1.9%, structural_LR=3.2%

Raw log:
```text
Command has been aborted
```
GT:
```text
Command has been aborted
```
Sample0:
```text
Command has been aborted
```

Target-batch parsed outputs:
- Sample5: `Command has been aborted`
- Sample6: `Command has been aborted`
- Sample7: `Command has been aborted`
- Sample8: `Command has been aborted.`
- Sample9: `Command has been aborted`

Nearest labeled references:
- Index28, FAIL=False: `Command has completed successfully`; GT=`Command has completed successfully`
- Index106, FAIL=False: `configured out`; GT=`configured out`
- Index46, FAIL=False: `update received`; GT=`update received`

Variable-free abort statement mostly copied correctly, with successful completed-command reference. This unseen TN shows useful cross-template generalization of a simple command family.


## C051: llama/disjoint/HPC/TN

ID `hpc-318493`, index293, seed0, slotextreme; seen GT=False. Risk33.888100, threshold199.452743, percentile12.39%, margin/IQR-1.3314. First/last votes0/5,3/5; failure label=0. Other method percentiles: Clotho=12.4%, U_published=78.4%, lexical_LR=75.4%, raw_hidden_LR=51.9%, same_space_LR=53.2%, same_space_LR_real_only=56.1%, same_space_two_density=34.2%, structural_LR=63.2%

Raw log:
```text
boot (command 1994) Error: Console-Busy Port already in use
```
GT:
```text
boot (command <*>) Error: Console-Busy Port already in use
```
Sample0:
```text
boot (<*> command <*> ) Error: <*>-<> Port already in use
```

Target-batch parsed outputs:
- Sample5: `boot (command <*>) Error: Console-Busy Port already in use`
- Sample6: `boot (command <*> ) Error: Console-Busy Port already in use`
- Sample7: `boot (command <*>) Error: Console-Busy Port already in use`
- Sample8: `boot (<*> ) Error: <*> Port already in use`
- Sample9: `boot (command <*>) Error: Console-Busy Port already in use`

Nearest labeled references:
- Index35, FAIL=True: `boot (command 2836) Error: node went to SRM while trying to boot`; GT=`boot (command <*>) Error: node went to SRM while trying to boot`
- Index23, FAIL=True: `boot (command 2142) Error: node went to SRM while trying to boot`; GT=`boot (command <*>) Error: node went to SRM while trying to boot`
- Index7, FAIL=True: `boot (command 2690) Error: node went to SRM while trying to boot`; GT=`boot (command <*>) Error: node went to SRM while trying to boot`

Same GT as C029 but both batches reverse from 0/5 to 3/5 success. Both receive low risk; contrasting outcomes expose uncertainty that a deterministic input predictor cannot perfectly distinguish.


## C052: llama/disjoint/Zookeeper/TN

ID `zookeeper-50589`, index384, seed0, slotnear_threshold; seen GT=False. Risk-36.615035, threshold-36.337216, percentile41.49%, margin/IQR-0.0090. First/last votes4/5,4/5; failure label=0. Other method percentiles: Clotho=41.5%, U_published=20.2%, lexical_LR=31.3%, raw_hidden_LR=11.3%, same_space_LR=35.5%, same_space_LR_real_only=32.1%, same_space_two_density=37.7%, structural_LR=50.2%

Raw log:
```text
Revalidating client: 0x24f0557806a0022
```
GT:
```text
Revalidating client: <*>
```
Sample0:
```text
Revalidating client: <*>
```

Target-batch parsed outputs:
- Sample5: `Revalidating client: <*>`
- Sample6: `Revalidating client: <*>`
- Sample7: `Revalidating client: <*>`
- Sample8: `Revalidating client: <*>`
- Sample9: `Revalidating client: <*>.`

Nearest labeled references:
- Index237, FAIL=True: `Got zxid 0xc00000001 expected 0xb00000085`; GT=`Got zxid <*> expected <*>`
- Index190, FAIL=True: `Got zxid 0xb0000007c expected 0xb00000085`; GT=`Got zxid <*> expected <*>`
- Index466, FAIL=False: `Getting a diff from the leader 0xd0000001b`; GT=`Getting a diff from the leader <*>`

Hex client ID mostly masked correctly despite a failing zxid neighbor. Just-below-threshold TN is not proof of calibrated confidence or a precise hexadecimal rule.


## C053: llama/disjoint/Hadoop/TN

ID `hadoop-13`, index108, seed0, slotnear_threshold; seen GT=False. Risk193.261952, threshold195.619264, percentile65.00%, margin/IQR-0.0141. First/last votes5/5,5/5; failure label=0. Other method percentiles: Clotho=65.0%, U_published=19.8%, lexical_LR=53.8%, raw_hidden_LR=22.2%, same_space_LR=27.4%, same_space_LR_real_only=29.4%, same_space_two_density=63.4%, structural_LR=58.6%

Raw log:
```text
mapreduce.task.io.sort.mb: 100
```
GT:
```text
mapreduce.task.io.sort.mb: <*>
```
Sample0:
```text
mapreduce.task.io.sort.mb: <*>
```

Target-batch parsed outputs:
- Sample5: `mapreduce.task.io.sort.mb: <*>`
- Sample6: `mapreduce.task.io.sort.mb: <*>`
- Sample7: `mapreduce.task.io.sort.mb: <*>`
- Sample8: `mapreduce.task.io.sort.mb: <*>`
- Sample9: `mapreduce.task.io.sort.mb: <*>`

Nearest labeled references:
- Index62, FAIL=True: `Using ShuffleConsumerPlugin: org.apache.hadoop.mapreduce.task.reduce.Shuffle@1fcaf332`; GT=`Using ShuffleConsumerPlugin: <*>`
- Index83, FAIL=False: `Web app /mapreduce started at 57855`; GT=`Web app <*> started at <*>`
- Index162, FAIL=True: `Using ShuffleConsumerPlugin: org.apache.hadoop.mapreduce.task.reduce.Shuffle@38fe9315`; GT=`Using ShuffleConsumerPlugin: <*>`

All ten correctly mask a configuration value despite a failing shuffle-plugin neighbor. Near-threshold TN counters the claim that semantic nearest-neighbor label alone determines the score.


## C054: llama/disjoint/Mac/TN

ID `mac-8718`, index375, seed0, slotseeded_representative; seen GT=False. Risk-23.009532, threshold8.941744, percentile0.60%, margin/IQR-1.1759. First/last votes3/5,3/5; failure label=0. Other method percentiles: Clotho=0.6%, U_published=1.4%, lexical_LR=0.2%, raw_hidden_LR=7.8%, same_space_LR=9.8%, same_space_LR_real_only=5.8%, same_space_two_density=2.2%, structural_LR=0.6%

Raw log:
```text
BuildActDeviceEntry enter
```
GT:
```text
BuildActDeviceEntry enter
```
Sample0:
```text
BuildActDeviceEntry enter
```

Target-batch parsed outputs:
- Sample5: `BuildActDeviceEntry enter`
- Sample6: `BuildAct<*>`
- Sample7: `BuildAct<*>`
- Sample8: `BuildActDeviceEntry enter`
- Sample9: `BuildActDeviceEntry enter`

Nearest labeled references:
- Index459, FAIL=False: `BuildActDeviceEntry exit`; GT=`BuildActDeviceEntry exit`
- Index464, FAIL=True: `vm_compressor_flush - starting`; GT=`vm_compressor_flush - starting`
- Index473, FAIL=True: `Peripheral manager is not powered on`; GT=`Peripheral manager is not powered on`

Entry/exit paired phrasing has a passing neighbor and majority correct copying. A genuinely unseen template can still belong to a familiar successful family.


## C055: llama/disjoint/Spark/TN

ID `spark-13020533`, index243, seed0, slotseeded_representative; seen GT=False. Risk61.043231, threshold89.276598, percentile22.29%, margin/IQR-0.1288. First/last votes4/5,4/5; failure label=0. Other method percentiles: Clotho=22.3%, U_published=21.1%, lexical_LR=35.5%, raw_hidden_LR=32.7%, same_space_LR=17.9%, same_space_LR_real_only=12.2%, same_space_two_density=29.1%, structural_LR=9.8%

Raw log:
```text
Started 6 remote fetches in 113 ms
```
GT:
```text
Started <*> remote fetches in <*> ms
```
Sample0:
```text
Started <*> remote fetches in <*> ms
```

Target-batch parsed outputs:
- Sample5: `Started <*> remote fetches in <*> ms`
- Sample6: `Started <*> remote fetches in <*> ms`
- Sample7: `Started <*> remote fetches in <*> ms`
- Sample8: `Started <*> remote <*> in <*> ms`
- Sample9: `Started <*> remote fetches in <*> ms`

Nearest labeled references:
- Index188, FAIL=True: `MemoryStore started with capacity 17.7 GB`; GT=`MemoryStore started with capacity <*>`
- Index63, FAIL=True: `MemoryStore started with capacity 27.8 GB`; GT=`MemoryStore started with capacity <*>`
- Index431, FAIL=True: `Exception in task 0.1 in stage 2.0 (TID 82)`; GT=`Exception in task <*> in stage <*> (TID <*>)`

Remote-fetch numeric fields mostly correct despite a failing memory-capacity neighbor. Low risk can generalize successfully without close failure-label agreement.


## C056: llama/disjoint/HealthApp/TN

ID `healthapp-105344`, index426, seed0, slotseeded_representative; seen GT=False. Risk-44.943183, threshold59.647218, percentile5.42%, margin/IQR-0.5470. First/last votes2/5,4/5; failure label=0. Other method percentiles: Clotho=5.4%, U_published=5.0%, lexical_LR=17.5%, raw_hidden_LR=16.7%, same_space_LR=19.5%, same_space_LR_real_only=21.5%, same_space_two_density=11.0%, structural_LR=15.7%

Raw log:
```text
calculateCaloriesWithCache totalCalories=60178
```
GT:
```text
calculateCaloriesWithCache totalCalories=<*>
```
Sample0:
```text
calculateCaloriesWithCache totalCalories=<*>
```

Target-batch parsed outputs:
- Sample5: `calculateCaloriesWithCache totalCalories=<*>`
- Sample6: `calculateCaloriesWithCache totalCalories=<*>`
- Sample7: `calculateCaloriesWithCache totalCalories=<*>`
- Sample8: `calculateCaloriesWithCache totalCalories=<*>`
- Sample9: `calculateCaloriesWithCache <*>`

Nearest labeled references:
- Index204, FAIL=False: `calculateAltitudeWithCache totalAltitude=540`; GT=`calculateAltitudeWithCache totalAltitude=<*>`
- Index169, FAIL=False: `calculateAltitudeWithCache totalAltitude=210`; GT=`calculateAltitudeWithCache totalAltitude=<*>`
- Index250, FAIL=False: `setBasicStandardSteps basicStandardStep=0 restartSteps=3363`; GT=`setBasicStandardSteps basicStandardStep=<*> restartSteps=<*>`

Calories assignment mostly correct near successful altitude assignment. Observable key/value structure transfers beyond exact template, contrasting object-variable FN C005.


## C057: gemma/standard/OpenStack/FN

ID `openstack-173042`, index351, seed0, slotextreme; seen GT=True. Risk25.530418, threshold65.056541, percentile25.49%, margin/IQR-1.7136. First/last votes0/5,1/5; failure label=1. Other method percentiles: Clotho=25.5%, lexical_LR=74.8%, raw_hidden_LR=68.4%, same_space_LR=77.2%, same_space_LR_real_only=77.0%, same_space_two_density=74.8%, structural_LR=81.0%

Raw log:
```text
[instance: 59c23388-985d-423b-abbe-c3a2603f75a5] Deleting instance files /var/lib/nova/instances/59c23388-985d-423b-abbe-c3a2603f75a5_del
```
GT:
```text
[instance: <*>] Deleting instance files <*>
```
Sample0:
```text
[instance: <*>] Deleting instance files /var/lib/nova/instances/<*>
```

Target-batch parsed outputs:
- Sample5: `[instance: <*>] Deleting instance files /var/lib/nova/instances/<*>`
- Sample6: `[instance: <*>] Deleting instance files /var/lib/nova/instances/<*>`
- Sample7: `[instance: <*>] Deleting instance files /var/lib/nova/instances/<*>`
- Sample8: `[instance: <*>] Deleting instance files <*>`
- Sample9: `[instance: <*>] Deleting instance files /var/lib/nova/instances/<*>`

Nearest labeled references:
- Index83, FAIL=False: `[instance: 5e6a95a0-142a-4e6a-ab8e-6627c8ae0872] Deleting instance files /var/lib/nova/instances/5e6a95a0-142a-4e6a-ab8e-6627c8ae0872_del`; GT=`[instance: <*>] Deleting instance files <*>`
- Index136, FAIL=True: `[instance: 29d7b230-75ab-4140-81d8-7353d8f7e69b] Deleting instance files /var/lib/nova/instances/29d7b230-75ab-4140-81d8-7353d8f7e69b_del`; GT=`[instance: <*>] Deleting instance files <*>`
- Index195, FAIL=True: `[instance: b3718a37-1a9f-48df-89f7-8c42661ec3a8] Deleting instance files /var/lib/nova/instances/b3718a37-1a9f-48df-89f7-8c42661ec3a8_del`; GT=`[instance: <*>] Deleting instance files <*>`

Path prefix retention causes 9/10 failures, yet a successful same-template neighbor and low score favor success. H1 explains geometry but not reliable failure ranking; compare C078's successful instance family.


## C058: gemma/standard/OpenStack/FN

ID `openstack-19408`, index499, seed0, slotextreme; seen GT=True. Risk29.018199, threshold65.056541, percentile38.90%, margin/IQR-1.5624. First/last votes0/5,1/5; failure label=1. Other method percentiles: Clotho=38.9%, lexical_LR=72.1%, raw_hidden_LR=73.3%, same_space_LR=74.5%, same_space_LR_real_only=76.5%, same_space_two_density=74.5%, structural_LR=82.3%

Raw log:
```text
[instance: 63a0d960-70b6-44c6-b606-491478a5cadf] Deleting instance files /var/lib/nova/instances/63a0d960-70b6-44c6-b606-491478a5cadf_del
```
GT:
```text
[instance: <*>] Deleting instance files <*>
```
Sample0:
```text
[instance: <*>] Deleting instance files /var/lib/nova/instances/<*>
```

Target-batch parsed outputs:
- Sample5: `[instance: <*>] Deleting instance files /var/lib/nova/instances/<*>`
- Sample6: `[instance: <*>] Deleting instance files /var/lib/nova/instances/<*>`
- Sample7: `[instance: <*>] Deleting instance files /var/lib/nova/instances/<*>`
- Sample8: `[instance: <*>] Deleting instance files /var/lib/nova/instances/<*>  _del`
- Sample9: `[instance: <*>] Deleting instance files <*>`

Nearest labeled references:
- Index83, FAIL=False: `[instance: 5e6a95a0-142a-4e6a-ab8e-6627c8ae0872] Deleting instance files /var/lib/nova/instances/5e6a95a0-142a-4e6a-ab8e-6627c8ae0872_del`; GT=`[instance: <*>] Deleting instance files <*>`
- Index136, FAIL=True: `[instance: 29d7b230-75ab-4140-81d8-7353d8f7e69b] Deleting instance files /var/lib/nova/instances/29d7b230-75ab-4140-81d8-7353d8f7e69b_del`; GT=`[instance: <*>] Deleting instance files <*>`
- Index195, FAIL=True: `[instance: b3718a37-1a9f-48df-89f7-8c42661ec3a8] Deleting instance files /var/lib/nova/instances/b3718a37-1a9f-48df-89f7-8c42661ec3a8_del`; GT=`[instance: <*>] Deleting instance files <*>`

Second deletion-path line repeats C057's failure mechanism and same successful neighbor. Same-family label heterogeneity matters; two examples do not establish prevalence independently.


## C059: gemma/standard/Spark/FN

ID `spark-11356`, index350, seed0, slotnear_threshold; seen GT=False. Risk259.183228, threshold259.687847, percentile39.39%, margin/IQR-0.0022. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=39.4%, lexical_LR=9.1%, raw_hidden_LR=3.5%, same_space_LR=14.3%, same_space_LR_real_only=5.0%, same_space_two_density=11.6%, structural_LR=23.8%

Raw log:
```text
Asked to remove non-existent executor 1
```
GT:
```text
Asked to remove non-existent executor <*>
```
Sample0:
```text
Asked to remove non-existent <*> <*>
```

Target-batch parsed outputs:
- Sample5: `Asked to remove non-existent <*>`
- Sample6: `Asked to remove non-existent <*> <*>`
- Sample7: `Asked to remove non-existent <*> 1`
- Sample8: `Asked to remove non-existent <*> <*>`
- Sample9: `Asked to remove non-existent <*> <*>`

Nearest labeled references:
- Index481, FAIL=False: `Requesting to kill executor(s) 1`; GT=`Requesting to kill executor(s) <*>`
- Index213, FAIL=False: `Updating epoch to 102 and clearing cache`; GT=`Updating epoch to <*> and clearing cache`
- Index253, FAIL=False: `Got assigned task 151786`; GT=`Got assigned task <*>`

All ten overmask executor noun in a simple command. Successful kill-executor neighbor and near-threshold risk show structural similarity need not preserve which words are constant.


## C060: gemma/standard/BGL/FN

ID `bgl-3457671`, index324, seed0, slotnear_threshold; seen GT=False. Risk262.318955, threshold262.943179, percentile52.56%, margin/IQR-0.0035. First/last votes1/5,0/5; failure label=1. Other method percentiles: Clotho=52.6%, lexical_LR=84.3%, raw_hidden_LR=53.8%, same_space_LR=77.7%, same_space_LR_real_only=64.5%, same_space_two_density=69.6%, structural_LR=54.6%

Raw log:
```text
uncorrectable error detected in external DDR...0
```
GT:
```text
uncorrectable error detected in external DDR...<*>
```
Sample0:
```text
uncorrectable error detected in external DDR...<*>
```

Target-batch parsed outputs:
- Sample5: `uncorrectable error detected in external <*>.`
- Sample6: `uncorrectable error detected in external <*>...0`
- Sample7: `uncorrectable error detected in external <*>...0`
- Sample8: `uncorrectable error detected in external <*>... <*>`
- Sample9: `uncorrectable error detected in external DDR<*>...<*>`

Nearest labeled references:
- Index222, FAIL=True: `uncorrectable error detected in directory 1....0`; GT=`uncorrectable error detected in directory <*>....<*>`
- Index497, FAIL=False: `critical input interrupt enable...0`; GT=`critical input interrupt enable...<*>`
- Index278, FAIL=True: `correctable error detected in EDRAM bank 1.....0`; GT=`correctable error detected in EDRAM bank <*>`

Sample0 is exact while all five target generations fail DDR boundaries. Failing directory neighbor is available; inspect majority target rather than treating displayed sample0 as the label.


## C061: gemma/standard/Mac/FN

ID `mac-47773`, index142, seed0, slotseeded_representative; seen GT=False. Risk158.677673, threshold420.353911, percentile3.78%, margin/IQR-0.7615. First/last votes0/5,1/5; failure label=1. Other method percentiles: Clotho=3.8%, lexical_LR=17.9%, raw_hidden_LR=39.9%, same_space_LR=81.3%, same_space_LR_real_only=81.3%, same_space_two_density=3.3%, structural_LR=12.3%

Raw log:
```text
*** -[AVAssetResourceLoadingRequest finishLoading] was sent to an instance of AVAssetResourceLoadingRequest that was already finished. Ignoring.
```
GT:
```text
*** -[AVAssetResourceLoadingRequest finishLoading] was sent to an instance of <*> that was already finished. Ignoring.
```
Sample0:
```text
*** -[AVAssetResourceLoadingRequest *** finishLoading] was sent to an instance of AVAssetResourceLoadingRequest that was already finished. Ignoring.
```

Target-batch parsed outputs:
- Sample5: `*** -[AVAssetResourceLoadingRequest finishLoading] was sent to an instance of AVAssetResourceLoadingRequest that was already finished. Ignoring.`
- Sample6: `*** -[AVAssetResourceLoadingRequest <*> finishLoading] was sent to an instance of AVAssetResourceLoadingRequest that was already finished. Ignoring.`
- Sample7: `*** -[AVAssetResourceLoadingRequest <*>] was sent to an instance of AVAssetResourceLoadingRequest that was already finished. Ignoring.`
- Sample8: `*** -[AVAssetResourceLoadingRequest finishLoading] was sent to an instance of <*> that was already finished. Ignoring.`
- Sample9: `*** -[AVAssetResourceLoadingRequest finishLoading] was sent to an instance of AVAssetResourceLoadingRequest that was already finished. Ignoring.`

Nearest labeled references:
- Index77, FAIL=True: `<<<< MediaValidator >>>> mv_ValidateRFC4281CodecId: Unrecognized codec mp3.(null). Failed codec specific check.`; GT=`<<<< MediaValidator >>>> mv_ValidateRFC4281CodecId: Unrecognized codec <*>. Failed codec specific check.`
- Index272, FAIL=False: `NSFileSandboxingRequestRelatedItemExtension: an error was received from pboxd instead of a token. Domain: NSPOSIXErrorDomain, code: 1`; GT=`NSFileSandboxingRequestRelatedItemExtension: an error was received from pboxd instead of a token. Domain: <*>, code: <*>`
- Index496, FAIL=False: `**** [IOBluetoothFamily][ProcessBluetoothTransportShowsUpActionWL] -- calling IOBluetoothFamily's registerService() -- 0x5fd0 -- 0x9a00 -- 0x1800 ****`; GT=`**** [IOBluetoothFamily][ProcessBluetoothTransportShowsUpActionWL] -- calling IOBluetoothFamily's registerService() -- <*> -- <*> -- <*> ****`

Repeated class name in natural sentence is often kept rather than masked. Low-risk unseen long message has a failing codec neighbor; novelty and length do not by themselves force high risk.


## C062: gemma/standard/Hadoop/FN

ID `hadoop-436`, index115, seed0, slotseeded_representative; seen GT=True. Risk331.974609, threshold339.335736, percentile38.41%, margin/IQR-0.0250. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=38.4%, lexical_LR=64.8%, raw_hidden_LR=92.6%, same_space_LR=65.7%, same_space_LR_real_only=62.3%, same_space_two_density=64.5%, structural_LR=84.0%

Raw log:
```text
Added filter AM_PROXY_FILTER (class=org.apache.hadoop.yarn.server.webproxy.amfilter.AmIpFilter) to context static
```
GT:
```text
Added filter <*> (class=<*>) to context <*>
```
Sample0:
```text
Added filter <*> (class=***) to context static
```

Target-batch parsed outputs:
- Sample5: `Added filter <*> (class=<*>) to context static`
- Sample6: `Added filter <*> (class=<*> ) to context static`
- Sample7: `Added filter <*> (class=<*>) to context static`
- Sample8: `Added filter <*> (class=<*>) to context static`
- Sample9: `Added filter <*> (class=org.apache.hadoop.yarn.server.webproxy.amfilter.AmIpFilter) to context static`

Nearest labeled references:
- Index459, FAIL=True: `Added filter AM_PROXY_FILTER (class=org.apache.hadoop.yarn.server.webproxy.amfilter.AmIpFilter) to context mapreduce`; GT=`Added filter <*> (class=<*>) to context <*>`
- Index160, FAIL=False: `Error communicating with RM: Resource Manager doesn't recognize AttemptId: application_1445087491445_0007`; GT=`Error communicating with RM: Resource Manager doesn't recognize AttemptId: <*>`
- Index220, FAIL=False: `Using ShuffleConsumerPlugin: org.apache.hadoop.mapreduce.task.reduce.Shuffle@337d44e4`; GT=`Using ShuffleConsumerPlugin: <*>`

All ten keep context static or class path; same-template failing reference exists despite FN. Strong counterexample to exact-template familiarity being sufficient for prediction accuracy.


## C063: gemma/standard/Mac/FN

ID `mac-8780`, index287, seed0, slotseeded_representative; seen GT=False. Risk154.061065, threshold420.353911, percentile2.56%, margin/IQR-0.7750. First/last votes1/5,0/5; failure label=1. Other method percentiles: Clotho=2.6%, lexical_LR=4.8%, raw_hidden_LR=69.4%, same_space_LR=66.5%, same_space_LR_real_only=65.2%, same_space_two_density=0.1%, structural_LR=20.1%

Raw log:
```text
**** [BroadcomBluetoothHostController][SetupController] -- Delay HCI Reset by 300ms ****
```
GT:
```text
**** [BroadcomBluetoothHostController][SetupController] -- Delay HCI Reset by <*> ****
```
Sample0:
```text
**** [BroadcomBluetoothHostController][SetupController] -- Delay HCI Reset by <*>ms ****
```

Target-batch parsed outputs:
- Sample5: `****  [<*>][<*>]  -- Delay HCI Reset by <*>ms ****`
- Sample6: `**** [********][********] -- Delay HCI Reset by <*>ms ****`
- Sample7: `**** [<*>][<*>] -- Delay HCI Reset by <*>ms ****`
- Sample8: `**** [BroadcomBluetoothHostController][SetupController] -- Delay HCI Reset by <*>ms ****`
- Sample9: `**** [<*>][<*>] -- Delay HCI Reset by <*>ms ****`

Nearest labeled references:
- Index496, FAIL=False: `**** [IOBluetoothFamily][ProcessBluetoothTransportShowsUpActionWL] -- calling IOBluetoothFamily's registerService() -- 0x5fd0 -- 0x9a00 -- 0x1800 ****`; GT=`**** [IOBluetoothFamily][ProcessBluetoothTransportShowsUpActionWL] -- calling IOBluetoothFamily's registerService() -- <*> -- <*> -- <*> ****`
- Index77, FAIL=True: `<<<< MediaValidator >>>> mv_ValidateRFC4281CodecId: Unrecognized codec mp3.(null). Failed codec specific check.`; GT=`<<<< MediaValidator >>>> mv_ValidateRFC4281CodecId: Unrecognized codec <*>. Failed codec specific check.`
- Index404, FAIL=True: `-[NetworkAnalyticsEngine _writeJournalRecord:fromCellFingerprint:key:atLOI:ofKind:lqm:isFaulty:] Hashing of the primary key failed. Dropping the journal record.`; GT=`-[NetworkAnalyticsEngine _writeJournalRecord:fromCellFingerprint:key:atLOI:ofKind:lqm:isFaulty:] Hashing of the primary key failed. Dropping the journal record.`

All label-batch outputs retain ms or overmask bracketed constants. A successful Bluetooth-format neighbor may encourage low risk despite persistent boundary failure; compare correct timing abstraction C078.


## C064: gemma/standard/Mac/FP

ID `mac-47771`, index442, seed0, slotextreme; seen GT=False. Risk4321.321812, threshold420.353911, percentile99.88%, margin/IQR11.3527. First/last votes3/5,3/5; failure label=0. Other method percentiles: Clotho=99.9%, lexical_LR=88.2%, raw_hidden_LR=98.4%, same_space_LR=69.4%, same_space_LR_real_only=68.9%, same_space_two_density=99.9%, structural_LR=97.0%

Raw log:
```text
[15:45:00.925] HTTPRequest figHTTPDumpFinalURLAndRemoteAddress: address 2607:f8b0:401d::8 final url https://r2---sn-n4v7sn7l.c.docs.google.com/videoplayback?id=2e514773f43464a1&itag=22&source=picasa&requiressl=yes&ttl=transient&mm=30&mn=sn-n4v7sn7l&ms=nxu&mv=m&pl=32&ei=bRpcWciBEceg_APrnY7IBg&driveid=1gnwLnyKP9R-fC9jxSRZZwrozP28F1J5SDg&mime=video/mp4&lmt=1491688542449626&mt=1499208228&ip=2607:f140:6000:8:51cf:994c:36cb:1c56&ipbits=0&expire=1499222701&cp=QVJOQUNfUVdOQlhNOnlwZFotQjE0cHB4&sparams=ip,ipbits,expire,id,itag,source,requiressl,ttl,mm,mn,ms,mv,pl,ei,driveid,mime,lmt,cp&signature=4A681A1F0098A72657CBEA27CB92E1FA4E53698F.6756F013E02C3C0B007A44B04BBEB1013346BB50&key=ck2&app=explorer&cpn=Vt8eqXHZv-rwrYSf&c=WEB&cver=1.20170627
```
GT:
```text
[<*>] HTTPRequest figHTTPDumpFinalURLAndRemoteAddress: address <*> final url <*>
```
Sample0:
```text
[15:45:00.925] HTTPRequest figHTTPDumpFinalURLAndRemoteAddress: address <*> final url <*>
```

Target-batch parsed outputs:
- Sample5: `[<*>] HTTPRequest figHTTPDumpFinalURLAndRemoteAddress: address <*> final url <*>`
- Sample6: `[<*>] HTTPRequest figHTTPDumpFinalURLAndRemoteAddress: address <*> final url <*>`
- Sample7: `[<*>] HTTPRequest figHTTPDumpFinalURLAndRemoteAddress: address <*> final url <*>`
- Sample8: `[15:45:00.925] HTTPRequest figHTTPDumpFinalURLAndRemoteAddress: address <*> final url <*>`
- Sample9: `[15:45:00.925] HTTPRequest figHTTPDumpFinalURLAndRemoteAddress: address <*> final url https://r2---sn-n4v7sn7l.c.docs.google.com/videoplayback?id=2e514773f43464a1&itag=22&source=picasa&requiressl=yes&ttl=transient&mm=30&mn=sn-n4v7sn7l&ms=nxu&mv=m&pl=32&ei=bRpcWciBEceg_APrnY7IBg&driveid=1gnwLnyKP9R-fC9jxSRZZwrozP28F1J5SDg&mime=video/mp4&lmt=1491688542449626&mt=1499208228&ip=<*>&ipbits=0&expire=1499222701&cp=QVJOQUNfUVdOQlhNOnlwZFotQjE0cHB4&sparams=ip,ipbits,expire,id,itag,source,requiressl,ttl,mm,mn,ms,mv,pl,ei,driveid,mime,lmt,cp&signature=4A681A1F0098A72657CBEA27CB92E1FA4E53698F.6756F013E02C3C0B007A44B04BBEB1013346BB50&key=ck2&app=explorer&cpn=Vt8eqXHZv-rwrYSf&c=WEB&cver=1.20170627`

Nearest labeled references:
- Index144, FAIL=True: `[15:45:00.925] HTTPRequest figHttpRequestDidReceiveResponseCallback: received http response 403 (error -12660) for https://r2---sn-n4v7sn7l.c.docs.google.com/videoplayback?id=2e514773f43464a1&itag=22&source=picasa&requiressl=yes&ttl=transient&mm=30&mn=sn-n4v7sn7l&ms=nxu&mv=m&pl=32&ei=bRpcWciBEceg_APrnY7IBg&driveid=1gnwLnyKP9R-fC9jxSRZZwrozP28F1J5SDg&mime=video/mp4&lmt=1491688542449626&mt=1499208228&ip=2607:f140:6000:8:51cf:994c:36cb:1c56&ipbits=0&expire=1499222701&cp=QVJOQUNfUVdOQlhNOnlwZFotQjE0cHB4&sparams=ip,ipbits,expire,id,itag,source,requiressl,ttl,mm,mn,ms,mv,pl,ei,driveid,mime,lmt,cp&signature=4A681A1F0098A72657CBEA27CB92E1FA4E53698F.6756F013E02C3C0B007A44B04BBEB1013346BB50&key=ck2&app=explorer&cpn=Vt8eqXHZv-rwrYSf&c=WEB&cver=1.20170627`; GT=`[<*>] HTTPRequest figHttpRequestDidReceiveResponseCallback: received http response <*> (error <*>) for <*>`
- Index403, FAIL=True: `2017-07-07 01:47:43.565 GoogleSoftwareUpdateAgent[37644/0x7000002a0000] [lvl=2] -[KSOutOfProcessFetcher beginFetchWithDelegate:] KSOutOfProcessFetcher fetching from URL: "https://tools.google.com/service/update2?cup2hreq=96a77dd2c36b0c1f7363f7e0109259f6d9c4d7208696f2d432456adf72145f92&cup2key=7:4002560718"`; GT=`<*> <*> GoogleSoftwareUpdateAgent[<*>] [lvl=<*>] -[KSOutOfProcessFetcher beginFetchWithDelegate:] KSOutOfProcessFetcher fetching from URL: "<*>"`
- Index424, FAIL=True: `PBRequester failed with Error Error Domain=NSURLErrorDomain Code=-1009 "The Internet connection appears to be offline." UserInfo={NSUnderlyingError=0x7fb7f0026390 {Error Domain=kCFErrorDomainCFNetwork Code=-1009 "The Internet connection appears to be offline." UserInfo={NSErrorFailingURLStringKey=https://gs-loc.apple.com/clls/wloc, NSErrorFailingURLKey=https://gs-loc.apple.com/clls/wloc, _kCFStreamErrorCodeKey=8, _kCFStreamErrorDomainKey=12, NSLocalizedDescription=The Internet connection appears to be offline.}}, NSErrorFailingURLStringKey=https://gs-loc.apple.com/clls/wloc, NSErrorFailingURLKey=https://gs-loc.apple.com/clls/wloc, _kCFStreamErrorDomainKey=12, _kCFStreamErrorCodeKey=8, NSLocalizedDescription=The Internet connection appears to be offline.}`; GT=`PBRequester failed with Error Error Domain=<*> Code=<*> <*>" <*> {Error Domain=<*> Code=<*> <*>" <*> <*> _kCFStreamErrorCodeKey=<*>, _kCFStreamErrorDomainKey=<*>, NSLocalizedDescription=<*>}}, <*> <*> _kCFStreamErrorDomainKey=<*>, _kCFStreamErrorCodeKey=<*>, NSLocalizedDescription=<*>}`

Long URL/timestamp record is mostly correct but high-risk; 3/5 in each batch means instability remains. Nearly same URL in a failing different message suggests content similarity may conflate task conventions.


## C065: gemma/standard/Zookeeper/FP

ID `zookeeper-72632`, index432, seed0, slotextreme; seen GT=True. Risk555.818727, threshold114.578954, percentile85.73%, margin/IQR6.3804. First/last votes0/5,3/5; failure label=0. Other method percentiles: Clotho=85.7%, lexical_LR=71.1%, raw_hidden_LR=21.6%, same_space_LR=97.9%, same_space_LR_real_only=95.5%, same_space_two_density=84.8%, structural_LR=77.1%

Raw log:
```text
Follower sid: 2 : info : org.apache.zookeeper.server.quorum.QuorumPeer$QuorumServer@9736e99
```
GT:
```text
Follower sid: <*> : info : <*>
```
Sample0:
```text
Follower sid: <*> : info : org.apache.zookeeper.server.quorum.QuorumPeer$QuorumServer@<*>
```

Target-batch parsed outputs:
- Sample5: `Follower sid: <*> : info : <*>`
- Sample6: `Follower sid: <*> : info : <*>`
- Sample7: `Follower sid: <*> : info : org.apache.zookeeper.server.quorum.QuorumPeer$QuorumServer@<*>`
- Sample8: `Follower sid: <*> : info : <*><*>`
- Sample9: `Follower sid: <*> : info : <*>`

Nearest labeled references:
- Index194, FAIL=True: `Follower sid: 2 : info : org.apache.zookeeper.server.quorum.QuorumPeer$QuorumServer@9317ccb`; GT=`Follower sid: <*> : info : <*>`
- Index423, FAIL=False: `Synchronizing with Follower sid: 1 maxCommittedLog=0x100001564 minCommittedLog=0x100001370 peerLastZxid=0x100001564`; GT=`Synchronizing with Follower sid: <*> maxCommittedLog=<*> minCommittedLog=<*> peerLastZxid=<*>`
- Index487, FAIL=True: `Exception causing close of session 0x14ed93111f20011 due to java.io.IOException: Connection reset by peer`; GT=`Exception causing close of session <*> due to java.io.IOException: Connection reset by peer`

First batch all fail but last batch majority succeeds on object identity. Failing same-family neighbor and high score fit H3 and family-risk transfer, not a stable FP explanation alone.


## C066: gemma/standard/HDFS/FP

ID `hdfs-8700259`, index274, seed0, slotnear_threshold; seen GT=True. Risk42.790134, threshold42.580999, percentile51.59%, margin/IQR0.0010. First/last votes2/5,4/5; failure label=0. Other method percentiles: Clotho=51.6%, lexical_LR=11.8%, raw_hidden_LR=54.5%, same_space_LR=43.5%, same_space_LR_real_only=32.8%, same_space_two_density=48.2%, structural_LR=29.8%

Raw log:
```text
PacketResponder blk_9160796703302881264 1 Exception java.io.IOException: Connection reset by peer
```
GT:
```text
PacketResponder <*> <*> Exception java.io.IOException: Connection reset by peer
```
Sample0:
```text
PacketResponder <*> 1 Exception java.io.IOException: <*>
```

Target-batch parsed outputs:
- Sample5: `PacketResponder <*> <*> Exception java.io.IOException: Connection reset by peer`
- Sample6: `PacketResponder <*> <*> Exception java.io.IOException: Connection reset by peer`
- Sample7: `PacketResponder <*> 1 Exception java.io.IOException: Connection reset by peer`
- Sample8: `PacketResponder <*> <*> Exception java.io.IOException: Connection reset by peer`
- Sample9: `PacketResponder <*> <*> Exception java.io.IOException: Connection reset by peer`

Nearest labeled references:
- Index16, FAIL=True: `PacketResponder blk_-6059290274455367169 2 Exception java.io.IOException: Connection reset by peer`; GT=`PacketResponder <*> <*> Exception java.io.IOException: Connection reset by peer`
- Index306, FAIL=False: `writeBlock blk_-3126359724254586349 received exception java.io.IOException: Connection reset by peer`; GT=`writeBlock <*> received exception java.io.IOException: Connection reset by peer`
- Index431, FAIL=True: `Exception in receiveBlock for block blk_-4723951162006187997 java.io.IOException: Connection reset by peer`; GT=`Exception in receiveBlock for block <*> java.io.IOException: Connection reset by peer`

Responder index retained in some generations; first/last majority flips. Same-GT failing neighbor and threshold FP contrast TP C074, exposing stochastic rather than categorical structural distinction.


## C067: gemma/standard/HealthApp/FP

ID `healthapp-3671`, index397, seed0, slotnear_threshold; seen GT=False. Risk63.344185, threshold63.065540, percentile47.93%, margin/IQR0.0048. First/last votes5/5,4/5; failure label=0. Other method percentiles: Clotho=47.9%, lexical_LR=61.1%, raw_hidden_LR=4.5%, same_space_LR=10.1%, same_space_LR_real_only=11.1%, same_space_two_density=27.9%, structural_LR=3.9%

Raw log:
```text
uploadStaticsToDB() onResult type = 0 obj=true
```
GT:
```text
uploadStaticsToDB() onResult type = <*> obj=<*>
```
Sample0:
```text
uploadStaticsToDB() onResult type = <*> obj=<*>
```

Target-batch parsed outputs:
- Sample5: `uploadStaticsToDB() onResult type = <*> obj=<*>`
- Sample6: `uploadStaticsToDB() onResult type = <*> obj=<*>`
- Sample7: `uploadStaticsToDB() onResult type = <*> obj=<*>`
- Sample8: `uploadStaticsToDB() onResult = <*> obj=<*>`
- Sample9: `uploadStaticsToDB() onResult type = <*> obj=<*>`

Nearest labeled references:
- Index419, FAIL=False: `saveRealTimeHealthDatasStat() size = 1,totalTime = 23`; GT=`saveRealTimeHealthDatasStat() size = <*>,totalTime = <*>`
- Index5, FAIL=False: `getMidWareStepCountService: isSystemApp = true`; GT=`getMidWareStepCountService: isSystemApp = <*>`
- Index50, FAIL=False: `subscribeHiHealthData() checkAppType 0 appID = 1`; GT=`subscribeHiHealthData() checkAppType <*> appID = <*>`

Nearly all numeric/boolean fields correctly abstracted, with a successful health-data neighbor. Near-threshold FP counters generic key/value difficulty.


## C068: gemma/standard/Hadoop/FP

ID `hadoop-128266`, index235, seed0, slotseeded_representative; seen GT=False. Risk439.923058, threshold339.335736, percentile56.22%, margin/IQR0.3418. First/last votes4/5,5/5; failure label=0. Other method percentiles: Clotho=56.2%, lexical_LR=52.6%, raw_hidden_LR=67.2%, same_space_LR=78.7%, same_space_LR_real_only=77.7%, same_space_two_density=35.5%, structural_LR=36.5%

Raw log:
```text
DefaultSpeculator.addSpeculativeAttempt -- we are speculating task_1445062781478_0012_m_000001
```
GT:
```text
DefaultSpeculator.addSpeculativeAttempt -- we are speculating <*>
```
Sample0:
```text
DefaultSpeculator.addSpeculativeAttempt -- we are speculating <*>.
```

Target-batch parsed outputs:
- Sample5: `DefaultSpeculator.addSpeculativeAttempt -- we are speculating <*>`
- Sample6: `DefaultSpeculator.addSpeculativeAttempt -- we are speculating <*>`
- Sample7: `DefaultSpeculator.addSpeculativeAttempt -- we are speculating <*>`
- Sample8: `DefaultSpeculator.addSpeculativeAttempt -- we are speculating <*>`
- Sample9: `DefaultSpeculator.addSpeculativeAttempt -- we are speculating <*>`

Nearest labeled references:
- Index296, FAIL=True: `Commit go/no-go request from attempt_1445182159119_0020_r_000000_0`; GT=`Commit go/no-go request from <*>`
- Index196, FAIL=False: `Task attempt_1445182159119_0014_r_000000_1000 is allowed to commit now`; GT=`Task <*> is allowed to commit now`
- Index396, FAIL=True: `Diagnostics report from attempt_1445062781478_0012_m_000001_0: Container released on a *lost* node`; GT=`Diagnostics report from <*>: Container released on a *lost* node`

Sample0 adds punctuation but all target generations succeed. Failing task-attempt neighbor accompanies FP; output-derived sample0 feature could disagree with majority labels.


## C069: gemma/standard/HDFS/FP

ID `hdfs-8713339`, index285, seed0, slotseeded_representative; seen GT=True. Risk274.707177, threshold42.580999, percentile79.88%, margin/IQR1.0925. First/last votes4/5,5/5; failure label=0. Other method percentiles: Clotho=79.9%, lexical_LR=66.0%, raw_hidden_LR=60.9%, same_space_LR=68.2%, same_space_LR_real_only=73.0%, same_space_two_density=79.9%, structural_LR=60.9%

Raw log:
```text
Exception in receiveBlock for block blk_6224343649004202692 java.net.SocketTimeoutException: 490000 millis timeout while waiting for channel to be ready for write. ch : java.nio.channels.SocketChannel[connected local=/10.251.127.243:44092 remote=/10.251.107.50:50010]
```
GT:
```text
Exception in receiveBlock for block <*> java.net.SocketTimeoutException: <*> millis timeout while waiting for channel to be ready for write. ch : java.nio.channels.SocketChannel[connected local=<*> remote=<*>]
```
Sample0:
```text
Exception in receiveBlock for block <*> java.net.SocketTimeoutException: <*> millis timeout while waiting for channel to be ready for write. ch : java.nio.channels.SocketChannel[connected local= <*> remote=<*>]
```

Target-batch parsed outputs:
- Sample5: `Exception in receiveBlock for block <*> java.net.SocketTimeoutException: <*> millis timeout while waiting for channel to be ready for write. ch : java.nio.channels.SocketChannel[connected local=<*> remote=<*>]`
- Sample6: `Exception in receiveBlock for block <*> java.net.SocketTimeoutException: <*> millis timeout while waiting for channel to be ready for write. ch : java.nio.channels.SocketChannel[connected local=<*> remote=<*>]`
- Sample7: `Exception in receiveBlock for block <*> java.net.SocketTimeoutException: <*> millis timeout while waiting for channel to be ready for write. ch : java.nio.channels.SocketChannel[connected local=<*> remote=<*>]`
- Sample8: `Exception in receiveBlock for block <*> java.net.SocketTimeoutException: <*> millis timeout while waiting for channel to be ready for write. ch : java.nio.channels.SocketChannel[connected local=<*> remote=<*>]`
- Sample9: `Exception in receiveBlock for block <*> java.net.SocketTimeoutException: <*> millis timeout while waiting for channel to be ready for write. ch : java.nio.channels.SocketChannel[connected local=<*> remote=<*>]`

Nearest labeled references:
- Index104, FAIL=True: `Exception in receiveBlock for block blk_584730932939516842 java.net.SocketTimeoutException: 485000 millis timeout while waiting for channel to be ready for write. ch : java.nio.channels.SocketChannel[connected local=/10.250.19.227:55473 remote=/10.251.107.50:50010]`; GT=`Exception in receiveBlock for block <*> java.net.SocketTimeoutException: <*> millis timeout while waiting for channel to be ready for write. ch : java.nio.channels.SocketChannel[connected local=<*> remote=<*>]`
- Index341, FAIL=True: `PacketResponder blk_-4723951162006187997 2 Exception java.net.SocketTimeoutException: 120000 millis timeout while waiting for channel to be ready for read. ch : java.nio.channels.SocketChannel[connected local=/10.251.107.98:46828 remote=/10.251.107.50:50010]`; GT=`PacketResponder <*> <*> Exception java.net.SocketTimeoutException: <*> millis timeout while waiting for channel to be ready for read. ch : java.nio.channels.SocketChannel[connected local=<*> remote=<*>]`
- Index95, FAIL=True: `writeBlock blk_-604967703652531741 received exception java.net.SocketTimeoutException: 60000 millis timeout while waiting for channel to be ready for read. ch : java.nio.channels.SocketChannel[connected local=/10.250.5.237:41159 remote=/10.251.39.160:50010]`; GT=`writeBlock <*> received exception java.net.SocketTimeoutException: <*> millis timeout while waiting for channel to be ready for <*> ch : java.nio.channels.SocketChannel[connected local=<*> remote=<*>]`

All five target generations correctly abstract timeout and socket endpoints although same-family reference fails. High risk is not proof that complex address records are intrinsically unparseable.


## C070: gemma/standard/Thunderbird/FP

ID `thunderbird-9428103`, index405, seed0, slotseeded_representative; seen GT=False. Risk500.513631, threshold196.909007, percentile90.12%, margin/IQR1.7170. First/last votes4/5,4/5; failure label=0. Other method percentiles: Clotho=90.1%, lexical_LR=24.3%, raw_hidden_LR=71.8%, same_space_LR=68.7%, same_space_LR_real_only=67.7%, same_space_two_density=89.4%, structural_LR=50.6%

Raw log:
```text
INFO: executing scheduler from '/var/spool/moab/' under UID 0 GID 0
```
GT:
```text
INFO: executing scheduler from <*> under UID <*> GID <*>
```
Sample0:
```text
INFO: executing scheduler from <*> under UID <*> GID <*>
```

Target-batch parsed outputs:
- Sample5: `INFO: executing scheduler from <*> under UID <*> GID <*>`
- Sample6: `INFO: executing scheduler from <*> under UID <*> GID <*>`
- Sample7: `INFO: executing scheduler from '/var/spool/moab/<*> under UID <*> GID <*>`
- Sample8: `INFO: executing scheduler from <*> under UID <*> GID <*>`
- Sample9: `INFO: executing scheduler from <*> under UID <*> GID <*>`

Nearest labeled references:
- Index249, FAIL=True: `fatal: the Postfix mail system is not running`; GT=`fatal: the Postfix mail system is not running`
- Index386, FAIL=False: `ACPI: PM-Timer IO Port: 0x808`; GT=`ACPI: PM-Timer IO Port: <*>`
- Index244, FAIL=False: `PCI: cache line size of 128 is not supported by device 0000:00:1d.7`; GT=`PCI: cache line size of <*> is not supported by device <*>`

Path quoting sometimes retained, but both batches majority succeed. A failing mail-status neighbor is only semantically broad; FP may reflect unfamiliarity rather than actual abstraction difficulty.


## C071: gemma/standard/OpenStack/TP

ID `openstack-157469`, index322, seed0, slotextreme; seen GT=True. Risk684.808931, threshold65.056541, percentile99.88%, margin/IQR26.8689. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=99.9%, lexical_LR=97.0%, raw_hidden_LR=81.1%, same_space_LR=99.4%, same_space_LR_real_only=97.7%, same_space_two_density=99.9%, structural_LR=96.2%

Raw log:
```text
10.11.10.1 "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1" status: 200 len: 1892 time: 0.2810302
```
GT:
```text
<*> "GET <*> <*>" status: <*> len: <*> time: <*>
```
Sample0:
```text
<*>"GET <*> HTTP/1.1" status: <*> len: <*> time: <*>
```

Target-batch parsed outputs:
- Sample5: `<*> "GET <*> HTTP/1.1" status: <*> len: <*> time: <*>`
- Sample6: `<*> "GET <*> HTTP/1.1" status: <*> len: <*> time: <*>`
- Sample7: `<*> "GET <*> HTTP/1.1" status: <*> len: <*> time: <*>`
- Sample8: `<*> "GET <*> HTTP/1.1" status: <*> len: <*> time: <*>`
- Sample9: `<*>"GET <*> HTTP/1.1" status: <*> len: <*> time: <*>`

Nearest labeled references:
- Index256, FAIL=True: `10.11.10.1 "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1" status: 200 len: 1759 time: 0.2559509`; GT=`<*> "GET <*> <*>" status: <*> len: <*> time: <*>`
- Index117, FAIL=True: `10.11.10.1 "POST /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers HTTP/1.1" status: 202 len: 733 time: 0.5107961`; GT=`<*> "POST <*> <*>" status: <*> len: <*> time: <*>`
- Index139, FAIL=True: `10.11.10.1 "POST /v2/e9746973ac574c6b8a9e8857f56a7608/os-server-external-events HTTP/1.1" status: 200 len: 380 time: 0.1037860`; GT=`<*> "POST <*> <*>" status: <*> len: <*> time: <*>`

All ten retain HTTP/1.1 where GT masks it. Same-family failing reference and extremely high score support family-level correctness transfer; compare low-risk deletion-path FN C057.


## C072: gemma/standard/OpenStack/TP

ID `openstack-56112`, index247, seed0, slotextreme; seen GT=True. Risk677.871565, threshold65.056541, percentile99.63%, margin/IQR26.5681. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=99.6%, lexical_LR=97.4%, raw_hidden_LR=84.5%, same_space_LR=97.2%, same_space_LR_real_only=97.4%, same_space_two_density=99.6%, structural_LR=96.2%

Raw log:
```text
10.11.10.1 "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1" status: 200 len: 1892 time: 0.2474380
```
GT:
```text
<*> "GET <*> <*>" status: <*> len: <*> time: <*>
```
Sample0:
```text
<*> "GET <*> HTTP/1.1" status: <*> len: <*> time: <*>
```

Target-batch parsed outputs:
- Sample5: `<*> "GET <*> HTTP/1.1" status: <*> len: <*> time: <*>`
- Sample6: `<*> "GET <*> HTTP/1.1" status: <*> len: <*> time: <*>`
- Sample7: `<*>"GET <*> HTTP/1.1" status: <*> len: <*> time: <*>`
- Sample8: `<*> "GET <*> HTTP/1.1" status: <*> len: <*> time: <*>`
- Sample9: `<*> "GET <*> HTTP/1.1" status: <*> len: <*> time: <*>`

Nearest labeled references:
- Index256, FAIL=True: `10.11.10.1 "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1" status: 200 len: 1759 time: 0.2559509`; GT=`<*> "GET <*> <*>" status: <*> len: <*> time: <*>`
- Index117, FAIL=True: `10.11.10.1 "POST /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers HTTP/1.1" status: 202 len: 733 time: 0.5107961`; GT=`<*> "POST <*> <*>" status: <*> len: <*> time: <*>`
- Index139, FAIL=True: `10.11.10.1 "POST /v2/e9746973ac574c6b8a9e8857f56a7608/os-server-external-events HTTP/1.1" status: 200 len: 380 time: 0.1037860`; GT=`<*> "POST <*> <*>" status: <*> len: <*> time: <*>`

Second HTTP access line repeats C071's retained-version error and high risk. Same-template examples are correlated evidence rather than independent mechanisms.


## C073: gemma/standard/Thunderbird/TP

ID `thunderbird-1014`, index236, seed0, slotnear_threshold; seen GT=False. Risk197.720069, threshold196.909007, percentile47.68%, margin/IQR0.0046. First/last votes1/5,0/5; failure label=1. Other method percentiles: Clotho=47.7%, lexical_LR=30.9%, raw_hidden_LR=1.6%, same_space_LR=5.7%, same_space_LR_real_only=5.2%, same_space_two_density=7.7%, structural_LR=11.8%

Raw log:
```text
using mwait in idle threads.
```
GT:
```text
using <*> in idle threads.
```
Sample0:
```text
using mwait in idle threads.
```

Target-batch parsed outputs:
- Sample5: `using mwait in idle threads.`
- Sample6: `using mwait in idle threads.`
- Sample7: `using mwait in idle threads.`
- Sample8: `using mwait in idle threads.`
- Sample9: `using mwait in idle threads.`

Nearest labeled references:
- Index23, FAIL=False: `SELinux: Disabled at runtime.`; GT=`SELinux: Disabled at runtime.`
- Index226, FAIL=False: `checking TSC synchronization across 2 CPUs: passed.`; GT=`checking TSC synchronization across <*> CPUs: passed.`
- Index164, FAIL=False: `loop: loaded (max 8 devices)`; GT=`loop: loaded (max <*> devices)`

The ordinary word mwait is retained in every target generation though GT masks it. Near-threshold TP illustrates word-like variable difficulty can be detected without a matching family.


## C074: gemma/standard/HDFS/TP

ID `hdfs-1379275`, index18, seed0, slotnear_threshold; seen GT=True. Risk43.704174, threshold42.580999, percentile51.83%, margin/IQR0.0053. First/last votes0/5,1/5; failure label=1. Other method percentiles: Clotho=51.8%, lexical_LR=17.2%, raw_hidden_LR=54.0%, same_space_LR=41.3%, same_space_LR_real_only=33.3%, same_space_two_density=51.8%, structural_LR=29.8%

Raw log:
```text
PacketResponder blk_7008279672769077211 1 Exception java.io.IOException: Connection reset by peer
```
GT:
```text
PacketResponder <*> <*> Exception java.io.IOException: Connection reset by peer
```
Sample0:
```text
PacketResponder <*> Exception <*>
```

Target-batch parsed outputs:
- Sample5: `PacketResponder <*> <*> Exception java.io.IOException: Connection reset by peer`
- Sample6: `PacketResponder <*> <*> Exception java.io.IOException: <*>`
- Sample7: `PacketResponder <*> 1 Exception java.io.IOException: Connection reset by peer`
- Sample8: `PacketResponder <*> <*> Exception java.io.IOException: <*>`
- Sample9: `PacketResponder <*> 1 Exception <*>: Connection reset by peer`

Nearest labeled references:
- Index16, FAIL=True: `PacketResponder blk_-6059290274455367169 2 Exception java.io.IOException: Connection reset by peer`; GT=`PacketResponder <*> <*> Exception java.io.IOException: Connection reset by peer`
- Index306, FAIL=False: `writeBlock blk_-3126359724254586349 received exception java.io.IOException: Connection reset by peer`; GT=`writeBlock <*> received exception java.io.IOException: Connection reset by peer`
- Index430, FAIL=False: `PacketResponder blk_9160796703302881264 2 Exception java.io.IOException: Broken pipe`; GT=`PacketResponder <*> <*> Exception java.io.IOException: Broken pipe`

Responder-index/exception-boundary variants yield only 1/5 successes. Score just above threshold contrasts FP C066 in same family; boundary labels are stochastic.


## C075: gemma/standard/HDFS/TP

ID `hdfs-8752054`, index478, seed0, slotseeded_representative; seen GT=False. Risk45.378564, threshold42.580999, percentile53.05%, margin/IQR0.0132. First/last votes2/5,2/5; failure label=1. Other method percentiles: Clotho=53.0%, lexical_LR=33.8%, raw_hidden_LR=58.2%, same_space_LR=50.9%, same_space_LR_real_only=45.7%, same_space_two_density=50.6%, structural_LR=32.1%

Raw log:
```text
PacketResponder blk_8006271611835981128 1 Exception java.io.IOException: The stream is closed
```
GT:
```text
PacketResponder <*> <*> Exception java.io.IOException: The stream is closed
```
Sample0:
```text
PacketResponder <*> java.io.IOException: The stream is closed
```

Target-batch parsed outputs:
- Sample5: `PacketResponder <*> 1 Exception java.io.IOException: The stream is closed`
- Sample6: `PacketResponder <*> Exception java.io.IOException: The stream is closed`
- Sample7: `PacketResponder <*> 1 Exception java.io.IOException: The stream is closed`
- Sample8: `PacketResponder <*> <*> Exception java.io.IOException: The stream is closed`
- Sample9: `PacketResponder <*> <*> Exception java.io.IOException: The stream is closed`

Nearest labeled references:
- Index16, FAIL=True: `PacketResponder blk_-6059290274455367169 2 Exception java.io.IOException: Connection reset by peer`; GT=`PacketResponder <*> <*> Exception java.io.IOException: Connection reset by peer`
- Index306, FAIL=False: `writeBlock blk_-3126359724254586349 received exception java.io.IOException: Connection reset by peer`; GT=`writeBlock <*> received exception java.io.IOException: Connection reset by peer`
- Index431, FAIL=True: `Exception in receiveBlock for block blk_-4723951162006187997 java.io.IOException: Connection reset by peer`; GT=`Exception in receiveBlock for block <*> java.io.IOException: Connection reset by peer`

Similar responder message has a different exception suffix and variable deletion. Both batches only 2/5 succeed; nearby failing reset variant is informative but not the same GT template.


## C076: gemma/standard/OpenSSH/TP

ID `openssh-221610`, index132, seed0, slotseeded_representative; seen GT=True. Risk443.621902, threshold33.808907, percentile71.10%, margin/IQR1.0250. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=71.1%, lexical_LR=87.7%, raw_hidden_LR=67.0%, same_space_LR=65.2%, same_space_LR_real_only=72.3%, same_space_two_density=74.3%, structural_LR=75.2%

Raw log:
```text
Received disconnect from 103.207.37.104: 11: Closed due to user request. [preauth]
```
GT:
```text
Received disconnect from <*>: <*>: Closed due to user request. [preauth]
```
Sample0:
```text
Received disconnect from <*>: <*> Closed due to user request. [preauth]
```

Target-batch parsed outputs:
- Sample5: `Received disconnect from <*>: <*> Closed due to user request. [preauth]`
- Sample6: `Received disconnect from <*>: <*> Closed due to user request. [preauth]`
- Sample7: `Received disconnect from <*>: <*> Closed due to user request. [preauth]`
- Sample8: `Received disconnect from <*>:  Closed due to user request. [preauth]`
- Sample9: `Received disconnect from <*>: <*>. Closed due to user request. [preauth]`

Nearest labeled references:
- Index496, FAIL=True: `Received disconnect from 103.207.37.179: 11: Closed due to user request. [preauth]`; GT=`Received disconnect from <*>: <*>: Closed due to user request. [preauth]`
- Index10, FAIL=True: `error: Received disconnect from 103.99.0.122: 14: No more user authentication methods available. [preauth]`; GT=`error: Received disconnect from <*>: <*>: No more user authentication methods available. [preauth]`
- Index99, FAIL=True: `Received disconnect from 137.189.204.253: 11: Disconnect requested by Windows SSH Client.`; GT=`Received disconnect from <*>: <*>: Disconnect requested by Windows SSH Client.`

Extra colon after disconnect reason is omitted in all ten. Same-family failure is correctly high-risk; punctuation failure is not restricted to unseen templates or FNs.


## C077: gemma/standard/BGL/TP

ID `bgl-4105393`, index301, seed0, slotseeded_representative; seen GT=False. Risk275.485912, threshold262.943179, percentile55.49%, margin/IQR0.0707. First/last votes0/5,1/5; failure label=1. Other method percentiles: Clotho=55.5%, lexical_LR=12.6%, raw_hidden_LR=37.4%, same_space_LR=48.4%, same_space_LR_real_only=62.8%, same_space_two_density=70.1%, structural_LR=32.2%

Raw log:
```text
plus zero................................0
```
GT:
```text
plus zero................................<*>
```
Sample0:
```text
plus <*>................................<*>
```

Target-batch parsed outputs:
- Sample5: `plus <*>................................<*>`
- Sample6: `plus zero................................<*>`
- Sample7: `plus <*>................................<*>`
- Sample8: `plus <*>................................<*>`
- Sample9: `plus <*>................................<*>`

Nearest labeled references:
- Index498, FAIL=True: `plus infinity............................0`; GT=`plus infinity............................<*>`
- Index264, FAIL=False: `correctable error.....1`; GT=`correctable error.....<*>`
- Index420, FAIL=True: `round toward +infinity...................0`; GT=`round toward +infinity...................<*>`

The word zero is overmasked while trailing numeric field should be masked. A failing infinity neighbor and modest high score fit transfer of formatting difficulty.


## C078: gemma/standard/OpenStack/TN

ID `openstack-122572`, index354, seed0, slotextreme; seen GT=True. Risk15.661744, threshold65.056541, percentile0.85%, margin/IQR-2.1415. First/last votes5/5,4/5; failure label=0. Other method percentiles: Clotho=0.9%, lexical_LR=4.0%, raw_hidden_LR=8.7%, same_space_LR=12.3%, same_space_LR_real_only=11.6%, same_space_two_density=37.7%, structural_LR=34.6%

Raw log:
```text
[instance: 1fc45bad-79db-463b-b18a-d293800f29bb] Took 20.83 seconds to build instance.
```
GT:
```text
[instance: <*>] Took <*> seconds to build instance.
```
Sample0:
```text
[instance: <*>] Took <*> seconds to build instance.
```

Target-batch parsed outputs:
- Sample5: `[instance: <*>] Took <*> seconds to build instance.`
- Sample6: `[instance: <*>] Took <*> to build instance.`
- Sample7: `[instance: <*>] Took <*> seconds to build instance.`
- Sample8: `[instance: <*>] Took <*> seconds to build instance.`
- Sample9: `[instance: <*>] Took <*> seconds to build instance.`

Nearest labeled references:
- Index469, FAIL=False: `[instance: 308efe9d-0616-473d-b7c3-606936bfabf6] Took 20.17 seconds to build instance.`; GT=`[instance: <*>] Took <*> seconds to build instance.`
- Index111, FAIL=False: `[instance: b2980ee5-ed52-4d81-933f-ed47b2595a73] Took 20.20 seconds to build instance.`; GT=`[instance: <*>] Took <*> seconds to build instance.`
- Index26, FAIL=False: `[instance: 8d64da06-29bb-466f-be46-5d8be6fbc34f] Took 20.28 seconds to build instance.`; GT=`[instance: <*>] Took <*> seconds to build instance.`

Instance/timing sentence mostly abstracted correctly with passing same-template neighbor. Very low risk fits H1; compare path-boundary FN C057 within instance vocabulary.


## C079: gemma/standard/OpenStack/TN

ID `openstack-140708`, index444, seed0, slotextreme; seen GT=True. Risk15.907155, threshold65.056541, percentile1.10%, margin/IQR-2.1308. First/last votes4/5,5/5; failure label=0. Other method percentiles: Clotho=1.1%, lexical_LR=2.8%, raw_hidden_LR=7.7%, same_space_LR=6.2%, same_space_LR_real_only=6.5%, same_space_two_density=39.9%, structural_LR=34.6%

Raw log:
```text
[instance: 67e71ed2-e7f1-4494-8154-d5297a8adbce] Took 19.92 seconds to build instance.
```
GT:
```text
[instance: <*>] Took <*> seconds to build instance.
```
Sample0:
```text
[instance: <*>] Took <*> seconds to build instance.
```

Target-batch parsed outputs:
- Sample5: `[instance: <*>] Took <*> seconds to build instance.`
- Sample6: `[instance: <*>] Took <*> seconds to build instance.`
- Sample7: `[instance: <*>] Took <*> seconds to build instance.`
- Sample8: `[instance: <*>] Took <*> seconds to build instance.`
- Sample9: `[instance: <*>] Took <*> seconds to build instance.`

Nearest labeled references:
- Index469, FAIL=False: `[instance: 308efe9d-0616-473d-b7c3-606936bfabf6] Took 20.17 seconds to build instance.`; GT=`[instance: <*>] Took <*> seconds to build instance.`
- Index111, FAIL=False: `[instance: b2980ee5-ed52-4d81-933f-ed47b2595a73] Took 20.20 seconds to build instance.`; GT=`[instance: <*>] Took <*> seconds to build instance.`
- Index26, FAIL=False: `[instance: 8d64da06-29bb-466f-be46-5d8be6fbc34f] Took 20.28 seconds to build instance.`; GT=`[instance: <*>] Took <*> seconds to build instance.`

A second build-instance line is correct in all target generations. Correlated same-family TN supports reference familiarity but cannot isolate its causal contribution.


## C080: gemma/standard/Mac/TN

ID `mac-99002`, index160, seed0, slotnear_threshold; seen GT=False. Risk419.863585, threshold420.353911, percentile51.83%, margin/IQR-0.0014. First/last votes4/5,5/5; failure label=0. Other method percentiles: Clotho=51.8%, lexical_LR=1.8%, raw_hidden_LR=1.8%, same_space_LR=40.6%, same_space_LR_real_only=46.0%, same_space_two_density=65.0%, structural_LR=15.7%

Raw log:
```text
CoreAnimation: context hosting changed while locked!
```
GT:
```text
CoreAnimation: context hosting changed while locked!
```
Sample0:
```text
CoreAnimation: context hosting changed while locked!
```

Target-batch parsed outputs:
- Sample5: `CoreAnimation: context hosting changed while locked!`
- Sample6: `CoreAnimation: context hosting changed while locked!`
- Sample7: `CoreAnimation: context hosting changed while locked!`
- Sample8: `CoreAnimation: context hosting changed while locked!`
- Sample9: `CoreAnimation: context hosting changed while locked!`

Nearest labeled references:
- Index401, FAIL=False: `AppleThunderboltGenericHAL::earlyWake - complete - took 0 milliseconds`; GT=`AppleThunderboltGenericHAL::earlyWake - complete - took <*> milliseconds`
- Index402, FAIL=True: `en0::IO80211Interface::postMessage bssid changed`; GT=`<*>::<*>::<*> bssid changed`
- Index231, FAIL=False: `CoreAnimation: failed to send fence reply: 10000003`; GT=`CoreAnimation: failed to send fence reply: <*>`

Variable-free warning copied correctly in all ten; risk is just below threshold despite unrelated passing hardware neighbor. Unusual natural text can be easy to parse.


## C081: gemma/standard/Thunderbird/TN

ID `thunderbird-9428309`, index220, seed0, slotnear_threshold; seen GT=False. Risk196.573493, threshold196.909007, percentile46.95%, margin/IQR-0.0019. First/last votes0/5,3/5; failure label=0. Other method percentiles: Clotho=47.0%, lexical_LR=62.6%, raw_hidden_LR=52.8%, same_space_LR=72.1%, same_space_LR_real_only=70.6%, same_space_two_density=59.4%, structural_LR=38.5%

Raw log:
```text
MSchedProcessJobs(0,GlobalSQ,GlobalHQ)
```
GT:
```text
MSchedProcessJobs(<*>,<*>,<*>)
```
Sample0:
```text
MSchedProcessJobs(0,<*>,<*>).
```

Target-batch parsed outputs:
- Sample5: `MSchedProcessJobs(0,<*>,<*>)`
- Sample6: `MSchedProcessJobs(<*>,<*>,<*>)`
- Sample7: `MSchedProcessJobs(<*>,<*>,<*>)`
- Sample8: `MSchedProcessJobs(0,<*>,<*> )`
- Sample9: `MSchedProcessJobs(<*>,<*>,<*>)`

Nearest labeled references:
- Index180, FAIL=True: `MRMInitialize(NULL,EMsg,SC)`; GT=`MRMInitialize(<*>,<*>,<*>)`
- Index17, FAIL=True: `MSysProcessArgs(1,ArgV,FALSE)`; GT=`MSysProcessArgs(<*>,<*>,<*>)`
- Index474, FAIL=True: `PCI Interrupt Routing Table [\_SB_.PCI0.PBHI.PXB1._PRT]`; GT=`PCI Interrupt Routing Table [<*>]`

Function arguments switch from 0/5 to 3/5 success; retaining zero and punctuation explain unstable labels. Near-threshold TN should not be interpreted as stable correctness.


## C082: gemma/standard/Zookeeper/TN

ID `zookeeper-69668`, index406, seed0, slotseeded_representative; seen GT=True. Risk25.566834, threshold114.578954, percentile11.34%, margin/IQR-1.2871. First/last votes4/5,4/5; failure label=0. Other method percentiles: Clotho=11.3%, lexical_LR=17.0%, raw_hidden_LR=54.0%, same_space_LR=17.2%, same_space_LR_real_only=4.3%, same_space_two_density=20.1%, structural_LR=36.7%

Raw log:
```text
Getting a diff from the leader 0x100000082
```
GT:
```text
Getting a diff from the leader <*>
```
Sample0:
```text
Getting a diff from the leader <*>
```

Target-batch parsed outputs:
- Sample5: `Getting a diff from the leader <*>`
- Sample6: `Getting a diff from the leader <*>`
- Sample7: `Getting a diff from the leader <*>.`
- Sample8: `Getting a diff from the leader <*>`
- Sample9: `Getting a diff from the leader <*>`

Nearest labeled references:
- Index284, FAIL=False: `Getting a diff from the leader 0x100001564`; GT=`Getting a diff from the leader <*>`
- Index360, FAIL=False: `Getting a diff from the leader 0x0`; GT=`Getting a diff from the leader <*>`
- Index230, FAIL=True: `Send worker leaving thread`; GT=`Send worker leaving thread`

Hex leader position mostly masked correctly with same-family passing neighbor. Low risk fits familiar-success transfer and counters generic hex-failure hypothesis.


## C083: gemma/standard/Zookeeper/TN

ID `zookeeper-55`, index270, seed0, slotseeded_representative; seen GT=False. Risk75.817128, threshold114.578954, percentile53.78%, margin/IQR-0.5605. First/last votes4/5,5/5; failure label=0. Other method percentiles: Clotho=53.8%, lexical_LR=4.8%, raw_hidden_LR=15.2%, same_space_LR=57.7%, same_space_LR_real_only=57.7%, same_space_two_density=11.6%, structural_LR=24.8%

Raw log:
```text
Server environment:os.name=Linux
```
GT:
```text
Server environment:os.name=<*>
```
Sample0:
```text
Server environment:os.name=<*>
```

Target-batch parsed outputs:
- Sample5: `Server environment:os.name=<*>`
- Sample6: `Server environment:os.name=<*>`
- Sample7: `Server environment:os.name=<*>`
- Sample8: `Server environment:os.name=<*>`
- Sample9: `Server environment:os.name=<*>`

Nearest labeled references:
- Index283, FAIL=False: `Server environment:os.version=3.16.0-30-generic`; GT=`Server environment:os.version=<*>`
- Index102, FAIL=False: `Server environment:host.name=mesos-master-2`; GT=`Server environment:host.name=<*>`
- Index198, FAIL=False: `Server environment:os.arch=amd64`; GT=`Server environment:os.arch=<*>`

Unseen environment key is consistently correct near successful os.version family. Contrast C002: lexical family transfers but does not guarantee identical generation stability across models/keys.


## C084: gemma/standard/HDFS/TN

ID `hdfs-344129`, index340, seed0, slotseeded_representative; seen GT=True. Risk18.203546, threshold42.580999, percentile14.27%, margin/IQR-0.1147. First/last votes3/5,4/5; failure label=0. Other method percentiles: Clotho=14.3%, lexical_LR=54.8%, raw_hidden_LR=57.4%, same_space_LR=40.6%, same_space_LR_real_only=30.6%, same_space_two_density=10.6%, structural_LR=11.5%

Raw log:
```text
Received block blk_6309451058963891227 of size 67108864 from /10.251.215.192
```
GT:
```text
Received block <*> of size <*> from <*>
```
Sample0:
```text
Received block blk_ <*> of size <*> from <*>
```

Target-batch parsed outputs:
- Sample5: `Received block <*> of size <*> from <*>`
- Sample6: `Received block <*> of size <*> from <*>`
- Sample7: `Received block <*> of size <*> from <*>`
- Sample8: `Received block blk_* of size <*> from <*>`
- Sample9: `Received block <*> of size <*> from <*>`

Nearest labeled references:
- Index479, FAIL=False: `Received block blk_244609771404984724 of size 67108864 from /10.250.10.213`; GT=`Received block <*> of size <*> from <*>`
- Index314, FAIL=True: `Receiving block blk_-2087153416572551076 src: /10.251.31.5:33935 dest: /10.251.31.5:50010`; GT=`Receiving block <*> src: <*> dest: <*>`
- Index278, FAIL=True: `10.251.193.175:50010:Transmitted block blk_1558192129233782487 to /10.250.11.85:50010`; GT=`<*>:Transmitted block <*> to <*>`

Sample0 retains block prefix while last batch mostly correct; same-family passing reference and low risk fit majority label. One displayed generation is insufficient to call FN.


## C085: gemma/disjoint/Zookeeper/FN

ID `zookeeper-7`, index122, seed0, slotextreme; seen GT=False. Risk201.039280, threshold1162.903888, percentile13.83%, margin/IQR-2.9912. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=13.8%, U_published=92.1%, lexical_LR=66.6%, raw_hidden_LR=33.8%, same_space_LR=85.7%, same_space_LR_real_only=44.0%, same_space_two_density=33.8%, structural_LR=83.2%

Raw log:
```text
binding to port 0.0.0.0/0.0.0.0:2181
```
GT:
```text
binding to port <*>
```
Sample0:
```text
binding to port <*>/<*>:2181
```

Target-batch parsed outputs:
- Sample5: `binding to port <*>:<*>`
- Sample6: `binding to port <*>:<*>`
- Sample7: `binding to port <*>:<*>`
- Sample8: `binding to port <*>:<*>`
- Sample9: `binding to port <*><*>:2181`

Nearest labeled references:
- Index189, FAIL=True: `Unexpected exception, tries=0, connecting to /10.10.34.12:2888`; GT=`Unexpected exception, tries=<*>, connecting to <*>`
- Index393, FAIL=True: `Unexpected exception, tries=0, connecting to /10.10.34.13:2888`; GT=`Unexpected exception, tries=<*>, connecting to <*>`
- Index151, FAIL=True: `Expiring session 0x34ed9ac1c1e00bb, timeout of 10000ms exceeded`; GT=`Expiring session <*> timeout of <*> exceeded`

All ten retain port or split one address variable. Failing connection neighbor exists but very low risk misses it; familiar networking syntax can conceal failure from PASS-density score.


## C086: gemma/disjoint/Zookeeper/FN

ID `zookeeper-46`, index412, seed0, slotextreme; seen GT=False. Risk289.179543, threshold1162.903888, percentile22.34%, margin/IQR-2.7171. First/last votes2/5,0/5; failure label=1. Other method percentiles: Clotho=22.3%, U_published=43.2%, lexical_LR=52.1%, raw_hidden_LR=5.7%, same_space_LR=87.9%, same_space_LR_real_only=56.8%, same_space_two_density=33.4%, structural_LR=38.1%

Raw log:
```text
TCP NoDelay set to: true
```
GT:
```text
TCP NoDelay set to: <*>
```
Sample0:
```text
TCP NoDelay set to: true
```

Target-batch parsed outputs:
- Sample5: `TCP <*> set to: true`
- Sample6: `TCP <*> set to: <*>`
- Sample7: `TCP NoDelay set to: true`
- Sample8: `TCP <* set to: true`
- Sample9: `TCP NoDelay set to: true`

Nearest labeled references:
- Index198, FAIL=False: `Server environment:os.arch=amd64`; GT=`Server environment:os.arch=<*>`
- Index60, FAIL=False: `LEADING - LEADER ELECTION TOOK - 6255`; GT=`LEADING - LEADER ELECTION TOOK - <*>`
- Index240, FAIL=True: `autopurge.purgeInterval set to 0`; GT=`autopurge.purgeInterval set to <*>`

All target generations retain true or mask NoDelay, though reference environment assignment passes. Low risk conflates common key/value form with correct word-boundary choice.


## C087: gemma/disjoint/BGL/FN

ID `bgl-4105409`, index420, seed0, slotnear_threshold; seen GT=False. Risk61.231172, threshold61.783875, percentile24.60%, margin/IQR-0.0051. First/last votes5/5,2/5; failure label=1. Other method percentiles: Clotho=24.6%, U_published=47.4%, lexical_LR=32.4%, raw_hidden_LR=58.2%, same_space_LR=53.4%, same_space_LR_real_only=31.4%, same_space_two_density=41.8%, structural_LR=41.8%

Raw log:
```text
round toward +infinity...................0
```
GT:
```text
round toward +infinity...................<*>
```
Sample0:
```text
round toward +infinity...................<*>
```

Target-batch parsed outputs:
- Sample5: `<*>`
- Sample6: `round toward +infinity...................<*>`
- Sample7: `round toward <*>....................<*>`
- Sample8: `round toward +infinity...................<*>`
- Sample9: `round toward <*>....................<*>`

Nearest labeled references:
- Index498, FAIL=True: `plus infinity............................0`; GT=`plus infinity............................<*>`
- Index30, FAIL=False: `memory manager miscompare........................0`; GT=`memory manager miscompare........................<*>`
- Index303, FAIL=False: `i-cache parity error..............0`; GT=`i-cache parity error..............<*>`

First batch all succeeds while last only 2/5 succeed through overmasking infinity. Near-threshold FN fits H3; a failing numeric-format neighbor does not make this stable failure.


## C088: gemma/disjoint/Spark/FN

ID `spark-13418508`, index273, seed0, slotnear_threshold; seen GT=False. Risk305.843750, threshold308.896894, percentile22.69%, margin/IQR-0.0078. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=22.7%, U_published=77.7%, lexical_LR=66.9%, raw_hidden_LR=93.4%, same_space_LR=72.1%, same_space_LR_real_only=60.8%, same_space_two_density=25.5%, structural_LR=84.5%

Raw log:
```text
Doing the fetch; tracker endpoint = NettyRpcEndpointRef(spark://MapOutputTracker@10.10.34.30:34727)
```
GT:
```text
Doing the fetch; tracker endpoint = NettyRpcEndpointRef(spark://<*>)
```
Sample0:
```text
Doing the fetch; tracker endpoint = NettyRpcEndpointRef(spark://MapOutputTracker@<*>: <*>)
```

Target-batch parsed outputs:
- Sample5: `Doing the fetch; tracker endpoint = NettyRpcEndpointRef(spark://<*>:<*>)`
- Sample6: `Doing the fetch; tracker endpoint = NettyRpcEndpointRef(<*> )`
- Sample7: `Doing the fetch; tracker endpoint = NettyRpcEndpointRef(spark://MapOutputTracker@<*>:*)`
- Sample8: `Doing the fetch; tracker endpoint = NettyRpcEndpointRef(spark://<*>:<*>)`
- Sample9: `Doing the fetch; tracker endpoint = NettyRpcEndpointRef(spark://<*>:<*>)`

Nearest labeled references:
- Index415, FAIL=True: `Starting job: count at pnmf_dblp.py:420`; GT=`Starting job: count at <*>`
- Index441, FAIL=True: `Opening proxy : mesos-slave-17:41511`; GT=`Opening proxy : <*>`
- Index300, FAIL=False: `Still have 1 requests outstanding when connection from mesos-slave-20/10.10.34.30:41009 is closed`; GT=`Still have <*> requests outstanding when connection from <*> is closed`

All ten split endpoint content inconsistently within fixed wrapper. Failing Spark-job neighbor is loose; stable low-risk error remains after excluding uncertain labels.


## C089: gemma/disjoint/Spark/FN

ID `spark-6946333`, index292, seed0, slotseeded_representative; seen GT=False. Risk251.552477, threshold308.896894, percentile13.05%, margin/IQR-0.1470. First/last votes1/5,1/5; failure label=1. Other method percentiles: Clotho=13.1%, U_published=26.3%, lexical_LR=87.8%, raw_hidden_LR=28.3%, same_space_LR=38.0%, same_space_LR_real_only=41.6%, same_space_two_density=6.6%, structural_LR=11.8%

Raw log:
```text
Started reading broadcast variable 8929
```
GT:
```text
Started reading broadcast variable <*>
```
Sample0:
```text
Started reading broadcast variable <*>
```

Target-batch parsed outputs:
- Sample5: `Started reading broadcast <*> <*>`
- Sample6: `Started reading broadcast <*> <*>`
- Sample7: `Started reading broadcast <*> <*>`
- Sample8: `Started reading broadcast variable <*>`
- Sample9: `Started reading broadcast <*> <*>`

Nearest labeled references:
- Index253, FAIL=False: `Got assigned task 151786`; GT=`Got assigned task <*>`
- Index398, FAIL=False: `File Output Committer Algorithm version is 1`; GT=`File Output Committer Algorithm version is <*>`
- Index469, FAIL=False: `Cleaned RDD 117`; GT=`Cleaned RDD <*>`

Most generations mask the constant broadcast or lose variable. Successful numeric-task neighbor and low risk show simple numeric shape does not ensure constant preservation; sample0 is correct.


## C090: gemma/disjoint/OpenStack/FN

ID `openstack-180181`, index271, seed0, slotseeded_representative; seen GT=False. Risk306.950599, threshold833.057909, percentile75.51%, margin/IQR-1.1284. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=75.5%, U_published=74.3%, lexical_LR=84.0%, raw_hidden_LR=89.7%, same_space_LR=98.6%, same_space_LR_real_only=97.4%, same_space_two_density=75.1%, structural_LR=83.8%

Raw log:
```text
couldn't obtain the vcpu count from domain id: 5ca5dcc8-87ec-4b4a-8ded-6c9d555050e0, exception: Requested operation is not valid: cpu affinity is not supported
```
GT:
```text
couldn't obtain the vcpu count from domain id: <*> exception: Requested operation is not valid: cpu affinity is not supported
```
Sample0:
```text
couldn't obtain the vcpu count from domain id: <*>, exception: <*>
```

Target-batch parsed outputs:
- Sample5: `couldn't obtain the vcpu count from domain id: <*>, exception: <*>`
- Sample6: `couldn't obtain the vcpu count from domain id: <*>, exception: <*>`
- Sample7: `couldn't obtain the vcpu count from domain id: <*>, exception: <*>`
- Sample8: `couldn't obtain the vcpu count from domain id: <*>, exception: <*>`
- Sample9: `couldn't obtain the vcpu count from domain id: <*>, exception: <*>`

Nearest labeled references:
- Index490, FAIL=True: `[instance: 65d8375a-4bc7-4706-b5fe-1049ba733851] Deletion of /var/lib/nova/instances/65d8375a-4bc7-4706-b5fe-1049ba733851_del complete`; GT=`[instance: <*>] Deletion of <*> complete`
- Index479, FAIL=True: `[instance: 657e9e59-0d1a-4043-b413-5f8912e6fecb] Deletion of /var/lib/nova/instances/657e9e59-0d1a-4043-b413-5f8912e6fecb_del complete`; GT=`[instance: <*>] Deletion of <*> complete`
- Index349, FAIL=False: `[instance: 05aab6c8-0298-43a3-b439-102d389a632d] disk limit not specified, defaulting to unlimited`; GT=`[instance: <*>] disk limit not specified, defaulting to unlimited`

All ten overmask a constant exception explanation and keep a comma absent from GT. Population rank is already 76th percentile but system threshold is higher; this FN is not globally low-ranked.


## C091: gemma/disjoint/Thunderbird/FN

ID `thunderbird-152`, index495, seed0, slotseeded_representative; seen GT=False. Risk311.642722, threshold459.242468, percentile32.20%, margin/IQR-0.5306. First/last votes2/5,1/5; failure label=1. Other method percentiles: Clotho=32.2%, U_published=47.4%, lexical_LR=74.6%, raw_hidden_LR=50.2%, same_space_LR=23.4%, same_space_LR_real_only=32.6%, same_space_two_density=34.2%, structural_LR=73.4%

Raw log:
```text
[ib_sm_sweep.c:1482]: No configuration change required
```
GT:
```text
[<*>]: No configuration change required
```
Sample0:
```text
[<*>]: No configuration change required
```

Target-batch parsed outputs:
- Sample5: `[<*>]: No configuration change required`
- Sample6: `[<*>]: <*>`
- Sample7: `[<*]: <*>`
- Sample8: `[<*>]: <*>`
- Sample9: `[<*>]: <*>`

Nearest labeled references:
- Index324, FAIL=True: `Unmounting PanFS filesystems: succeeded`; GT=`Unmounting <*> filesystems: <*>`
- Index219, FAIL=True: `hw_random: RNG not detected`; GT=`hw_random: RNG not detected`
- Index90, FAIL=True: `panfs mount: Dribble Writes mode is enabled`; GT=`panfs mount: <*> mode is enabled`

Bracketed filename line changes from 2/5 to 1/5 success; exact sample0 hides overmasking in target batch. Low risk cannot be explained from one displayed generation alone.


## C092: gemma/disjoint/Mac/FP

ID `mac-47771`, index442, seed0, slotextreme; seen GT=False. Risk7026.165297, threshold820.337930, percentile97.40%, margin/IQR9.0940. First/last votes3/5,3/5; failure label=0. Other method percentiles: Clotho=97.4%, U_published=98.2%, lexical_LR=85.8%, raw_hidden_LR=68.6%, same_space_LR=96.2%, same_space_LR_real_only=96.2%, same_space_two_density=97.4%, structural_LR=75.0%

Raw log:
```text
[15:45:00.925] HTTPRequest figHTTPDumpFinalURLAndRemoteAddress: address 2607:f8b0:401d::8 final url https://r2---sn-n4v7sn7l.c.docs.google.com/videoplayback?id=2e514773f43464a1&itag=22&source=picasa&requiressl=yes&ttl=transient&mm=30&mn=sn-n4v7sn7l&ms=nxu&mv=m&pl=32&ei=bRpcWciBEceg_APrnY7IBg&driveid=1gnwLnyKP9R-fC9jxSRZZwrozP28F1J5SDg&mime=video/mp4&lmt=1491688542449626&mt=1499208228&ip=2607:f140:6000:8:51cf:994c:36cb:1c56&ipbits=0&expire=1499222701&cp=QVJOQUNfUVdOQlhNOnlwZFotQjE0cHB4&sparams=ip,ipbits,expire,id,itag,source,requiressl,ttl,mm,mn,ms,mv,pl,ei,driveid,mime,lmt,cp&signature=4A681A1F0098A72657CBEA27CB92E1FA4E53698F.6756F013E02C3C0B007A44B04BBEB1013346BB50&key=ck2&app=explorer&cpn=Vt8eqXHZv-rwrYSf&c=WEB&cver=1.20170627
```
GT:
```text
[<*>] HTTPRequest figHTTPDumpFinalURLAndRemoteAddress: address <*> final url <*>
```
Sample0:
```text
[15:45:00.925] HTTPRequest figHTTPDumpFinalURLAndRemoteAddress: address <*> final url <*>
```

Target-batch parsed outputs:
- Sample5: `[<*>] HTTPRequest figHTTPDumpFinalURLAndRemoteAddress: address <*> final url <*>`
- Sample6: `[<*>] HTTPRequest figHTTPDumpFinalURLAndRemoteAddress: address <*> final url <*>`
- Sample7: `[<*>] HTTPRequest figHTTPDumpFinalURLAndRemoteAddress: address <*> final url <*>`
- Sample8: `[15:45:00.925] HTTPRequest figHTTPDumpFinalURLAndRemoteAddress: address <*> final url <*>`
- Sample9: `[15:45:00.925] HTTPRequest figHTTPDumpFinalURLAndRemoteAddress: address <*> final url https://r2---sn-n4v7sn7l.c.docs.google.com/videoplayback?id=2e514773f43464a1&itag=22&source=picasa&requiressl=yes&ttl=transient&mm=30&mn=sn-n4v7sn7l&ms=nxu&mv=m&pl=32&ei=bRpcWciBEceg_APrnY7IBg&driveid=1gnwLnyKP9R-fC9jxSRZZwrozP28F1J5SDg&mime=video/mp4&lmt=1491688542449626&mt=1499208228&ip=<*>&ipbits=0&expire=1499222701&cp=QVJOQUNfUVdOQlhNOnlwZFotQjE0cHB4&sparams=ip,ipbits,expire,id,itag,source,requiressl,ttl,mm,mn,ms,mv,pl,ei,driveid,mime,lmt,cp&signature=4A681A1F0098A72657CBEA27CB92E1FA4E53698F.6756F013E02C3C0B007A44B04BBEB1013346BB50&key=ck2&app=explorer&cpn=Vt8eqXHZv-rwrYSf&c=WEB&cver=1.20170627`

Nearest labeled references:
- Index144, FAIL=True: `[15:45:00.925] HTTPRequest figHttpRequestDidReceiveResponseCallback: received http response 403 (error -12660) for https://r2---sn-n4v7sn7l.c.docs.google.com/videoplayback?id=2e514773f43464a1&itag=22&source=picasa&requiressl=yes&ttl=transient&mm=30&mn=sn-n4v7sn7l&ms=nxu&mv=m&pl=32&ei=bRpcWciBEceg_APrnY7IBg&driveid=1gnwLnyKP9R-fC9jxSRZZwrozP28F1J5SDg&mime=video/mp4&lmt=1491688542449626&mt=1499208228&ip=2607:f140:6000:8:51cf:994c:36cb:1c56&ipbits=0&expire=1499222701&cp=QVJOQUNfUVdOQlhNOnlwZFotQjE0cHB4&sparams=ip,ipbits,expire,id,itag,source,requiressl,ttl,mm,mn,ms,mv,pl,ei,driveid,mime,lmt,cp&signature=4A681A1F0098A72657CBEA27CB92E1FA4E53698F.6756F013E02C3C0B007A44B04BBEB1013346BB50&key=ck2&app=explorer&cpn=Vt8eqXHZv-rwrYSf&c=WEB&cver=1.20170627`; GT=`[<*>] HTTPRequest figHttpRequestDidReceiveResponseCallback: received http response <*> (error <*>) for <*>`
- Index240, FAIL=True: `CCFile::copyFile fileName is [2017-07-03_16,27,05.017683]-AirPortBrcm4360_Logs-040.txt, source path:/var/log/CoreCapture/com.apple.driver.AirPort.Brcm4360.0/DriverLogs//[2017-07-03_16,27,05.017683]-AirPortBrcm4360_Logs-040.txt, dest path:/Library/Logs/CrashReporter/CoreCapture/com.apple.driver.AirPort.Brcm4360.0/[2017-07-03_16,27,05.123034]=AuthFail:sts:5_rsn:0/DriverLogs//[2017-07-03_16,27,05.017683]-AirPortBrcm4360_Logs-040.txt`; GT=`CCFile::copyFile fileName is [<*>]-<*>, source path:<*>, dest path:<*>`
- Index383, FAIL=True: `2017-07-07 17:01:58.333 GoogleSoftwareUpdateAgent[38375/0x7000002a0000] [lvl=2] +[KSCodeSigningVerification verifyBundle:applicationId:error:] KSCodeSigningVerification verifying code signing for '/Users/xpc/Library/Google/GoogleSoftwareUpdate/GoogleSoftwareUpdate.bundle/Contents/MacOS/GoogleSoftwareUpdateDaemon' with the requirement 'anchor apple generic and certificate 1[field.1.2.840.113635.100.6.2.6] exists and certificate leaf[field.1.2.840.113635.100.6.1.13] exists and certificate leaf[subject.OU]="EQHXZ8M8AV" and (identifier="com.google.Keystone")'`; GT=`<*> <*> GoogleSoftwareUpdateAgent[<*>] [lvl=<*>] +[KSCodeSigningVerification verifyBundle:applicationId:error:] KSCodeSigningVerification verifying code signing for <*> with the requirement 'anchor apple generic and certificate <*>[field.<*>] exists and certificate leaf[field.<*>] exists and certificate leaf[subject.OU]=<*> and (identifier=<*>)'`

Same long URL input as C064 remains an FP in disjoint data. An almost identical URL in a failing response message makes content-similarity confounding plausible, while batch instability persists.


## C093: gemma/disjoint/Thunderbird/FP

ID `thunderbird-15905`, index203, seed0, slotextreme; seen GT=False. Risk1838.616609, threshold459.242468, percentile99.40%, margin/IQR4.9586. First/last votes5/5,5/5; failure label=0. Other method percentiles: Clotho=99.4%, U_published=26.2%, lexical_LR=68.2%, raw_hidden_LR=37.0%, same_space_LR=46.6%, same_space_LR_real_only=28.6%, same_space_two_density=97.4%, structural_LR=33.0%

Raw log:
```text
you want, please write a subnet declaration
```
GT:
```text
you want, please write a subnet declaration
```
Sample0:
```text
you want, please write a subnet declaration
```

Target-batch parsed outputs:
- Sample5: `you want, please write a subnet declaration`
- Sample6: `you want, please write a subnet declaration`
- Sample7: `you want, please write a subnet declaration`
- Sample8: `you want, please write a subnet declaration`
- Sample9: `you want, please write a subnet declaration`

Nearest labeled references:
- Index429, FAIL=False: `stopping the Postfix mail system`; GT=`stopping the Postfix mail system`
- Index5, FAIL=True: `Kickstart: setup the cisco (topspin) ib stack`; GT=`Kickstart: setup the cisco (topspin) ib stack`
- Index295, FAIL=False: `Loading keyring`; GT=`Loading keyring`

Variable-free sentence copied correctly in all ten yet 99th-percentile risk. Strong unfamiliar-success counterexample to novelty-as-failure; compare header TP C099/C100.


## C094: gemma/disjoint/Linux/FP

ID `linux-33`, index296, seed0, slotnear_threshold; seen GT=False. Risk602.574995, threshold602.287953, percentile71.00%, margin/IQR0.0007. First/last votes4/5,5/5; failure label=0. Other method percentiles: Clotho=71.0%, U_published=21.0%, lexical_LR=27.4%, raw_hidden_LR=57.4%, same_space_LR=59.8%, same_space_LR_real_only=54.6%, same_space_two_density=74.6%, structural_LR=44.8%

Raw log:
```text
Calibrating delay loop... 1441.79 BogoMIPS
```
GT:
```text
Calibrating delay loop... <*> BogoMIPS
```
Sample0:
```text
Calibrating delay loop... <*> BogoMIPS
```

Target-batch parsed outputs:
- Sample5: `Calibrating delay loop... <*> BogoMIPS`
- Sample6: `Calibrating delay loop... <*> BogoMIPS`
- Sample7: `Calibrating delay loop... <*> BogoMIPS`
- Sample8: `Calibrating delay loop... <*> BogoMIPS`
- Sample9: `Calibrating delay loop... <*> BogoMIPS`

Nearest labeled references:
- Index59, FAIL=True: `FDC 0 is a National Semiconductor PC87306`; GT=`FDC <*> is a National Semiconductor <*>`
- Index498, FAIL=False: `Started working: 30 available services`; GT=`Started working: <*> available services`
- Index344, FAIL=False: `inserting floppy driver for 2.6.5-1.358`; GT=`inserting floppy driver for <*>`

All target generations abstract BogoMIPS number correctly despite highish score and failing hardware neighbor. A single numeric boundary is not necessarily difficult.


## C095: gemma/disjoint/Mac/FP

ID `mac-93778`, index17, seed0, slotnear_threshold; seen GT=False. Risk825.436704, threshold820.337930, percentile59.80%, margin/IQR0.0075. First/last votes5/5,4/5; failure label=0. Other method percentiles: Clotho=59.8%, U_published=37.4%, lexical_LR=67.8%, raw_hidden_LR=51.4%, same_space_LR=14.6%, same_space_LR_real_only=12.6%, same_space_two_density=60.6%, structural_LR=62.6%

Raw log:
```text
ApplePersistenceIgnoreState: Existing state will not be touched. New state will be written to /var/folders/nt/fvzdv1fn7n1d5bpgqb_wcfxh0000gn/T/com.microsoft.Word/com.microsoft.Word.savedState
```
GT:
```text
ApplePersistenceIgnoreState: Existing state will not be touched. New state will be written to <*>
```
Sample0:
```text
ApplePersistenceIgnoreState: Existing state will not be touched. New state will be written to <*>
```

Target-batch parsed outputs:
- Sample5: `ApplePersistenceIgnoreState: Existing state will not be touched. New state will be written to <*>`
- Sample6: `ApplePersistenceIgnoreState: Existing state will not be touched. New state will be written to <*>/com.microsoft.Word/com.microsoft.Word.savedState`
- Sample7: `ApplePersistenceIgnoreState: Existing state will not be touched. New state will be written to <*>`
- Sample8: `ApplePersistenceIgnoreState: Existing state will not be touched. New state will be written to <*>`
- Sample9: `ApplePersistenceIgnoreState: Existing state will not be touched. New state will be written to <*>`

Nearest labeled references:
- Index338, FAIL=True: `realpath() returned NULL for /var/folders/zz/zyxvpxvq6csfxvn_n00000s0000068/C/com.apple.SoftwareUpdate`; GT=`realpath() returned <*> for <*>`
- Index201, FAIL=True: `network changed: v4(en0+:10.105.163.9) v6(en0:2607:f140:6000:8:c6b3:1ff:fecd:467f) DNS! Proxy SMB`; GT=`network changed: <*> DNS! Proxy SMB`
- Index227, FAIL=True: `Could not get event name for stream/token: com.apple.xpc.activity/4505: 132: Request for stale data`; GT=`Could not get event name for <*>: <*>: <*>: Request for stale data`

Long destination path mostly masked as one slot while reference path error fails. Near-threshold FP cautions against inferring harm from path presence or long raw input.


## C096: gemma/disjoint/OpenSSH/FP

ID `openssh-633185`, index262, seed0, slotseeded_representative; seen GT=False. Risk600.554906, threshold232.697780, percentile76.51%, margin/IQR1.4208. First/last votes5/5,4/5; failure label=0. Other method percentiles: Clotho=76.5%, U_published=49.6%, lexical_LR=23.1%, raw_hidden_LR=48.4%, same_space_LR=56.8%, same_space_LR_real_only=61.6%, same_space_two_density=84.5%, structural_LR=87.3%

Raw log:
```text
Failed password for invalid user user1 from 181.214.87.4 port 36988 ssh2
```
GT:
```text
Failed password for invalid user <*> from <*> port <*> ssh2
```
Sample0:
```text
Failed password for invalid user <*> from <*> port <*> ssh2
```

Target-batch parsed outputs:
- Sample5: `Failed password for invalid user <*> from <*> port <*> ssh2`
- Sample6: `Failed password for invalid user <*> from <*> <*> ssh2`
- Sample7: `Failed password for invalid user <*> from <*> port <*> ssh2`
- Sample8: `Failed password for invalid user <*> from <*> port <*> ssh2`
- Sample9: `Failed password for invalid user <*> from <*> port <*> ssh2`

Nearest labeled references:
- Index82, FAIL=True: `Received disconnect from 45.32.81.34: 11: Bye Bye [preauth]`; GT=`Received disconnect from <*>: <*>: Bye Bye [preauth]`
- Index117, FAIL=True: `Received disconnect from 42.159.202.248: 11: Bye Bye [preauth]`; GT=`Received disconnect from <*>: <*>: Bye Bye [preauth]`
- Index462, FAIL=True: `Received disconnect from 179.170.204.170: 11: Bye Bye [preauth]`; GT=`Received disconnect from <*>: <*>: Bye Bye [preauth]`

Password-failure log mostly parsed correctly despite alarming semantics and failing disconnect neighbor. Detector should predict parsing correctness, not whether the log describes a system failure.


## C097: gemma/disjoint/HPC/FP

ID `hpc-13297`, index307, seed0, slotseeded_representative; seen GT=False. Risk3800.913970, threshold2191.028841, percentile69.12%, margin/IQR1.0162. First/last votes3/5,3/5; failure label=0. Other method percentiles: Clotho=69.1%, U_published=33.4%, lexical_LR=93.3%, raw_hidden_LR=54.0%, same_space_LR=73.7%, same_space_LR_real_only=64.5%, same_space_two_density=90.1%, structural_LR=92.4%

Raw log:
```text
risBoot (command 2847)
```
GT:
```text
risBoot (command <*>)
```
Sample0:
```text
risBoot (command <*>)
```

Target-batch parsed outputs:
- Sample5: `risBoot (command <*> )`
- Sample6: `risBoot (command <*>)`
- Sample7: `risBoot (command <*>)`
- Sample8: `risBoot (command <*>)`
- Sample9: `risBoot (command <*> )`

Nearest labeled references:
- Index66, FAIL=True: `boot (command 2607)`; GT=`boot (command <*>)`
- Index107, FAIL=True: `boot (command 4174)`; GT=`boot (command <*>)`
- Index385, FAIL=True: `haltcluster (command 2124)`; GT=`haltcluster (command <*>)`

Command-ID parentheses fluctuate only in spacing; majority succeeds in both batches near a failing boot family. Familiarity can transfer wrong difficulty across related commands.


## C098: gemma/disjoint/Thunderbird/FP

ID `thunderbird-10531975`, index496, seed0, slotseeded_representative; seen GT=False. Risk505.969320, threshold459.242468, percentile73.00%, margin/IQR0.1680. First/last votes4/5,5/5; failure label=0. Other method percentiles: Clotho=73.0%, U_published=56.2%, lexical_LR=44.2%, raw_hidden_LR=45.4%, same_space_LR=47.4%, same_space_LR_real_only=59.4%, same_space_two_density=73.4%, structural_LR=23.0%

Raw log:
```text
server nfs1 not responding, still trying
```
GT:
```text
server <*> not responding, still trying
```
Sample0:
```text
server <*> not responding, still <*>
```

Target-batch parsed outputs:
- Sample5: `server <*> not responding, still trying`
- Sample6: `server <*> not responding, still trying`
- Sample7: `server <*> not responding, still trying`
- Sample8: `server <*> not responding, still trying`
- Sample9: `server <*> not responding, still trying`

Nearest labeled references:
- Index112, FAIL=True: `sda: asking for cache data failed`; GT=`<*>: asking for cache data failed`
- Index272, FAIL=False: `Kickstart: convert to static ip on boot`; GT=`Kickstart: convert to static ip on boot`
- Index253, FAIL=False: `ACPI: Using IOAPIC for interrupt routing`; GT=`ACPI: Using IOAPIC for interrupt routing`

Target batch all correctly masks server name, although sample0 overmasks trying. Failing storage-error neighbor and FP demonstrate sample0 and aggregate labels differ.


## C099: gemma/disjoint/Hadoop/TP

ID `hadoop-4`, index127, seed0, slotextreme; seen GT=False. Risk8848.427948, threshold394.298861, percentile99.80%, margin/IQR31.5856. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=99.8%, U_published=89.8%, lexical_LR=30.2%, raw_hidden_LR=93.4%, same_space_LR=99.8%, same_space_LR_real_only=99.8%, same_space_two_density=99.8%, structural_LR=53.8%

Raw log:
```text
Executing with tokens:
```
GT:
```text
Executing with tokens:
```
Sample0:
```text
Executing with <*>
```

Target-batch parsed outputs:
- Sample5: `Executing with <*>`
- Sample6: `Executing with tokens: <*> <*> <*>`
- Sample7: `Executing with tokens: <*>`
- Sample8: `Executing with tokens: <*>`
- Sample9: `Executing with tokens: <*>`

Nearest labeled references:
- Index494, FAIL=True: `Setting job diagnostics to`; GT=`Setting job diagnostics to`
- Index197, FAIL=False: `JVM with ID : jvm_1445144423722_0023_m_000007 asked for a task`; GT=`JVM with ID : <*> asked for a task`
- Index305, FAIL=False: `Ramping up 1`; GT=`Ramping up <*>`

Variable-free heading is overmasked or completed in all ten, with maximal risk. Compare C093's stable success: Clotho captures some heading risk but overgeneralizes to other short text.


## C100: gemma/disjoint/BGL/TP

ID `bgl-4105372`, index245, seed0, slotextreme; seen GT=False. Risk2119.502958, threshold61.783875, percentile99.40%, margin/IQR19.1098. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=99.4%, U_published=91.0%, lexical_LR=9.6%, raw_hidden_LR=99.8%, same_space_LR=91.8%, same_space_LR_real_only=79.8%, same_space_two_density=99.4%, structural_LR=8.8%

Raw log:
```text
General Purpose Registers:
```
GT:
```text
General Purpose Registers:
```
Sample0:
```text
General Purpose Registers: <*>
```

Target-batch parsed outputs:
- Sample5: `General Purpose Registers:  <*> <*> <*> <*> <*>  <*> <*> <*>`
- Sample6: `General Purpose Registers: <*> <*> <*>`
- Sample7: `General Purpose Registers: <*>  <*> <*>`
- Sample8: `General Purpose Registers:  <*>  <*>  <*>  <*>  <*>  <*>  <*>`
- Sample9: `General Purpose Registers: <*> <*> <*>`

Nearest labeled references:
- Index149, FAIL=True: `Microloader Assertion`; GT=`Microloader Assertion`
- Index393, FAIL=True: `Starting SystemController`; GT=`Starting SystemController`
- Index379, FAIL=True: `problem state (0=sup,1=usr).......1`; GT=`problem state (<*>=sup,<*>=usr).......<*>`

Register heading receives invented variable payload in all ten. High score fits the continuation mechanism but C037/C093 show no universal variable-free rule.


## C101: gemma/disjoint/Hadoop/TP

ID `hadoop-403`, index376, seed0, slotnear_threshold; seen GT=False. Risk394.318334, threshold394.298861, percentile31.80%, margin/IQR0.0001. First/last votes3/5,1/5; failure label=1. Other method percentiles: Clotho=31.8%, U_published=74.6%, lexical_LR=16.6%, raw_hidden_LR=45.0%, same_space_LR=64.6%, same_space_LR_real_only=65.0%, same_space_two_density=54.6%, structural_LR=17.8%

Raw log:
```text
OutputCommitter set in config null
```
GT:
```text
OutputCommitter set in config <*>
```
Sample0:
```text
OutputCommitter set in config <*> null
```

Target-batch parsed outputs:
- Sample5: `OutputCommitter set in config <*> null`
- Sample6: `OutputCommitter set in config <*> null`
- Sample7: `OutputCommitter set in config <*> null`
- Sample8: `OutputCommitter set in config <*> null`
- Sample9: `OutputCommitter set in config <*>`

Nearest labeled references:
- Index326, FAIL=True: `ReduceTask metrics system started`; GT=`<*> metrics system started`
- Index303, FAIL=False: `Unable to parse prior job history, aborting recovery`; GT=`Unable to parse prior job history, aborting recovery`
- Index444, FAIL=True: `All maps assigned. Ramping up all remaining reduces:1`; GT=`All maps assigned. Ramping up all remaining reduces:<*>`

Null is alternately retained after placeholder; 3/5 versus 1/5 success flips target. Near-threshold TP is compatible with H3 and not evidence of strong margin separation.


## C102: gemma/disjoint/Spark/TP

ID `spark-14835912`, index357, seed0, slotnear_threshold; seen GT=False. Risk309.655697, threshold308.896894, percentile23.90%, margin/IQR0.0019. First/last votes2/5,1/5; failure label=1. Other method percentiles: Clotho=23.9%, U_published=93.8%, lexical_LR=54.0%, raw_hidden_LR=34.3%, same_space_LR=71.3%, same_space_LR_real_only=61.6%, same_space_two_density=31.5%, structural_LR=80.5%

Raw log:
```text
Ignored failure: java.io.IOException: Failed to send RPC 8212686680769530402 to mesos-slave-05/10.10.34.15:59498: java.nio.channels.ClosedChannelException
```
GT:
```text
Ignored failure: java.io.IOException: Failed to send RPC <*> to <*>: java.nio.channels.ClosedChannelException
```
Sample0:
```text
Ignored failure: java.io.IOException: Failed to send RPC <*> to <*">: java.nio.channels.ClosedChannelException
```

Target-batch parsed outputs:
- Sample5: `Ignored failure: java.io.IOException: Failed to send RPC <*> to mesos-slave-05/<*>: <*>: java.nio.channels.ClosedChannelException`
- Sample6: `Ignored failure: java.io.IOException: Failed to send RPC <*> to <*>: <*>: java.nio.channels.ClosedChannelException`
- Sample7: `Ignored failure: java.io.IOException: Failed to send RPC <*> to mesos-slave-<*>:<*>: java.nio.channels.ClosedChannelException`
- Sample8: `Ignored failure: java.io.IOException: Failed to send RPC <*> to <*><*>: java.nio.channels.ClosedChannelException`
- Sample9: `Ignored failure: java.io.IOException: Failed to send RPC <*> to <*>: java.nio.channels.ClosedChannelException`

Nearest labeled references:
- Index368, FAIL=False: `Lost task 1.0 in stage 5.0 (TID 9, mesos-slave-08): java.lang.OutOfMemoryError: Requested array size exceeds VM limit`; GT=`Lost task <*> in stage <*> (TID <*>, <*>): java.lang.OutOfMemoryError: Requested array size exceeds VM limit`
- Index165, FAIL=True: `Retrying connect to server: mesos-master-1/10.10.34.11:8030. Already tried 7 time(s); maxRetries=45`; GT=`Retrying connect to server: <*>. Already tried <*> time(s); maxRetries=<*>`
- Index274, FAIL=True: `Lost task 2.0 in stage 57.0 (TID 2435, mesos-slave-24): java.io.IOException: Cannot run program "/home/curi/anaconda2/bin/python": error=2, No such file or directory`; GT=`Lost task <*> in stage <*> (TID <*>, <*>): java.io.IOException: Cannot run program <*>: error=<*>, No such file or directory`

Compound destination and exception boundary variants mostly fail. Nearby task failure has a successful parsing label, yet score barely flags failure; neighbor label is not determinative.


## C103: gemma/disjoint/HPC/TP

ID `hpc-65275`, index70, seed0, slotseeded_representative; seen GT=False. Risk3211.770417, threshold2191.028841, percentile49.79%, margin/IQR0.6443. First/last votes0/5,2/5; failure label=1. Other method percentiles: Clotho=49.8%, U_published=64.9%, lexical_LR=18.7%, raw_hidden_LR=63.7%, same_space_LR=91.4%, same_space_LR_real_only=79.6%, same_space_two_density=73.7%, structural_LR=14.7%

Raw log:
```text
Link error on broadcast tree Interconnect-1T00:00:2:7
```
GT:
```text
Link error on broadcast tree <*>
```
Sample0:
```text
Link error on broadcast tree <*>: <*>
```

Target-batch parsed outputs:
- Sample5: `Link error on broadcast tree <*>`
- Sample6: `Link error on broadcast tree <*> <*>`
- Sample7: `Link error on broadcast tree <*>`
- Sample8: `Link error on broadcast tree <*><*>`
- Sample9: `Link error on broadcast tree <*><*>::<*>`

Nearest labeled references:
- Index93, FAIL=False: `ServerFileSystem: An ServerFileSystem domain panic has occurred on storage1432`; GT=`ServerFileSystem: An ServerFileSystem domain panic has occurred on <*>`
- Index150, FAIL=False: `ServerFileSystem: An ServerFileSystem domain panic has occurred on storage1223`; GT=`ServerFileSystem: An ServerFileSystem domain panic has occurred on <*>`
- Index417, FAIL=True: `Failed subcommands 2308\ 2315`; GT=`Failed subcommands <*>`

Broadcast-tree identifier is fragmented in three target generations; 2/5 succeed. Nearby passing storage-domain reference and TP counter a strict nearest-label account.


## C104: gemma/disjoint/Mac/TP

ID `mac-33208`, index378, seed0, slotseeded_representative; seen GT=False. Risk3902.012823, threshold820.337930, percentile93.40%, margin/IQR4.5159. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=93.4%, U_published=69.8%, lexical_LR=82.2%, raw_hidden_LR=69.8%, same_space_LR=43.4%, same_space_LR_real_only=56.2%, same_space_two_density=93.8%, structural_LR=72.2%

Raw log:
```text
[com.apple.calendar.store.log.subscription] [Failed to refresh <CalManagedSubscriptionInfo: 0x7fa11d1a7e80> (entity: SubscriptionInfo; id: 0x401eeb <x-coredata://DB05755C-483D-44B7-B93B-ED06E57FF420/SubscriptionInfo/p1> ; data: {
```
GT:
```text
[<*>] [Failed to refresh <CalManagedSubscriptionInfo: <*> (entity: SubscriptionInfo; id: <*> ; data: {
```
Sample0:
```text
[com.apple.calendar.store.log.subscription] [Failed to refresh <*> (entity: SubscriptionInfo; id: <*> ; data: {
```

Target-batch parsed outputs:
- Sample5: `[com.apple.calendar.store.log.subscription] [Failed to refresh <*>> (entity: SubscriptionInfo; id: <*>; data: {`
- Sample6: `[com.apple.calendar.store.log.subscription] [Failed to refresh <*> (entity: SubscriptionInfo; id: <*>  ; data: {`
- Sample7: `[com.apple.calendar.store.log.subscription] [Failed to refresh <*> (entity: SubscriptionInfo; id: <*> <*> ; data: {}`
- Sample8: `[com.apple.calendar.store.log.subscription] [Failed to refresh <*> (entity: SubscriptionInfo; id: <*> ; data: {`
- Sample9: `[com.apple.calendar.store.log.subscription] [Failed to refresh <*> (entity: <*>; id: <*> <*>; data: {`

Nearest labeled references:
- Index431, FAIL=True: `[com.apple.calendar.store.log.caldav.queue] [Account refresh failed with error: Error Domain=CoreDAVHTTPStatusErrorDomain Code=504 "(null)" UserInfo={AccountName=163, CalDAVErrFromRefresh=YES, CoreDAVHTTPHeaders=<CFBasicHash 0x7fa11af6e180 [0x7fff7abc1440]>{type = immutable dict, count = 5,`; GT=`[<*>] [Account refresh failed with error: Error Domain=CoreDAVHTTPStatusErrorDomain <*> "(<*>)" UserInfo={AccountName=<*>, CalDAVErrFromRefresh=<*>, CoreDAVHTTPHeaders=<*>{type = <*>, count = <*>,`
- Index8, FAIL=True: `2017-07-05 10:52:16.001 GoogleSoftwareUpdateAgent[35861/0x700000323000] [lvl=2] -[KSUpdateCheckAction performAction] KSUpdateCheckAction starting update check for ticket(s): {(`; GT=`<*> <*> GoogleSoftwareUpdateAgent[<*>] [lvl=<*>] -[KSUpdateCheckAction performAction] KSUpdateCheckAction starting update check for ticket(s): {(`
- Index180, FAIL=True: `2017-07-05 16:18:06.771 GoogleSoftwareUpdateAgent[36128/0x7000002a0000] [lvl=2] -[KSAgentApp uploadStats:] Successfully uploaded stats <KSStatsCollection:0x100315e20 path="/Users/xpc/Library/Google/GoogleSoftwareUpdate/Stats/Keystone.stats", count=5, stats={`; GT=`<*> <*> GoogleSoftwareUpdateAgent[<*>] [lvl=<*>] -[KSAgentApp uploadStats:] Successfully uploaded stats <KSStatsCollection:<*> path="<*>", count=<*>, stats={`

Long nested calendar object repeatedly retains subscription channel or corrupts brackets. Failing calendar reference and high risk fit cross-template structural transfer, contrasting stable low-risk C061.


## C105: gemma/disjoint/Thunderbird/TP

ID `thunderbird-10589077`, index411, seed0, slotseeded_representative; seen GT=False. Risk1274.343367, threshold459.242468, percentile97.00%, margin/IQR2.9301. First/last votes0/5,0/5; failure label=1. Other method percentiles: Clotho=97.0%, U_published=36.2%, lexical_LR=3.4%, raw_hidden_LR=44.6%, same_space_LR=30.6%, same_space_LR_real_only=5.8%, same_space_two_density=91.4%, structural_LR=49.8%

Raw log:
```text
Unable to handle kernel NULL pointer dereference at 00000000000000f0 RIP:
```
GT:
```text
Unable to handle kernel NULL pointer dereference at <*> RIP:
```
Sample0:
```text
Unable to handle kernel NULL pointer dereference at <*> RIP: <*>
```

Target-batch parsed outputs:
- Sample5: `Unable to handle kernel NULL pointer dereference at <*> RIP: <*>`
- Sample6: `Unable to handle kernel NULL pointer dereference at <*> RIP: <*>`
- Sample7: `Unable to handle kernel NULL pointer dereference at <*>> RIP:`
- Sample8: `Unable to handle kernel NULL pointer dereference at <*> RIP: <*>`
- Sample9: `Unable to handle kernel <*> dereference at <*> RIP:  <*>`

Nearest labeled references:
- Index485, FAIL=False: `frequency initialized 133.783 PPM from /var/lib/ntp/drift`; GT=`frequency initialized <*> PPM from <*>`
- Index2, FAIL=False: `Frequency format error in /var/lib/ntp/drift`; GT=`Frequency format error in <*>`
- Index70, FAIL=True: `[ib_sm_assign.c:389]: Failed to set LID on switch=66a100400017e, because NO RESPONSE`; GT=`[<*>]: Failed to set LID on switch=<*>, because NO RESPONSE`

Kernel header adds invented RIP variable in every target generation. High score despite a passing frequency neighbor shows useful risk information beyond nearest-label matching.


## C106: gemma/disjoint/Zookeeper/TN

ID `zookeeper-19117`, index91, seed0, slotextreme; seen GT=False. Risk37.996983, threshold1162.903888, percentile0.21%, margin/IQR-3.4982. First/last votes3/5,4/5; failure label=0. Other method percentiles: Clotho=0.2%, U_published=39.8%, lexical_LR=20.6%, raw_hidden_LR=79.4%, same_space_LR=36.0%, same_space_LR_real_only=27.0%, same_space_two_density=0.6%, structural_LR=24.5%

Raw log:
```text
FollowerRequestProcessor exited loop!
```
GT:
```text
FollowerRequestProcessor exited loop!
```
Sample0:
```text
FollowerRequestProcessor exited loop!
```

Target-batch parsed outputs:
- Sample5: `<*> exited loop!`
- Sample6: `FollowerRequestProcessor exited loop!`
- Sample7: `FollowerRequestProcessor exited loop!`
- Sample8: `FollowerRequestProcessor exited loop!`
- Sample9: `FollowerRequestProcessor exited loop!`

Nearest labeled references:
- Index347, FAIL=False: `CommitProcessor exited loop!`; GT=`CommitProcessor exited loop!`
- Index151, FAIL=True: `Expiring session 0x34ed9ac1c1e00bb, timeout of 10000ms exceeded`; GT=`Expiring session <*> timeout of <*> exceeded`
- Index144, FAIL=True: `Expiring session 0x34edfaa9c22000d, timeout of 10000ms exceeded`; GT=`Expiring session <*> timeout of <*> exceeded`

Follower/CommitProcessor exit-loop family is mostly copied correctly. Low risk generalizes successful family beyond exact template despite a few overmasking outputs.


## C107: gemma/disjoint/Zookeeper/TN

ID `zookeeper-19110`, index288, seed0, slotextreme; seen GT=False. Risk38.707184, threshold1162.903888, percentile0.64%, margin/IQR-3.4960. First/last votes5/5,5/5; failure label=0. Other method percentiles: Clotho=0.6%, U_published=26.2%, lexical_LR=20.2%, raw_hidden_LR=3.2%, same_space_LR=44.5%, same_space_LR_real_only=65.7%, same_space_two_density=13.4%, structural_LR=33.8%

Raw log:
```text
shutdown called
```
GT:
```text
shutdown called
```
Sample0:
```text
shutdown called
```

Target-batch parsed outputs:
- Sample5: `shutdown called`
- Sample6: `shutdown called`
- Sample7: `shutdown called`
- Sample8: `shutdown called`
- Sample9: `shutdown called`

Nearest labeled references:
- Index150, FAIL=False: `Shutdown called`; GT=`Shutdown called`
- Index413, FAIL=False: `Shutting down`; GT=`Shutting down`
- Index417, FAIL=False: `minSessionTimeout set to -1`; GT=`minSessionTimeout set to <*>`

Only capitalization distinguishes a passing reference Shutdown called from shutdown called. All ten correct and low risk illustrate exact-GT disjoint does not imply structural novelty.


## C108: gemma/disjoint/Spark/TN

ID `spark-13427369`, index308, seed0, slotnear_threshold; seen GT=False. Risk308.776493, threshold308.896894, percentile23.09%, margin/IQR-0.0003. First/last votes5/5,5/5; failure label=0. Other method percentiles: Clotho=23.1%, U_published=17.1%, lexical_LR=52.8%, raw_hidden_LR=10.2%, same_space_LR=1.0%, same_space_LR_real_only=2.6%, same_space_two_density=3.8%, structural_LR=61.2%

Raw log:
```text
Host added was in lost list earlier: mesos-slave-18
```
GT:
```text
Host added was in lost list earlier: <*>
```
Sample0:
```text
Host added was in lost list earlier: <*>
```

Target-batch parsed outputs:
- Sample5: `Host added was in lost list earlier: <*>`
- Sample6: `Host added was in lost list earlier: <*>`
- Sample7: `Host added was in lost list earlier: <*>`
- Sample8: `Host added was in lost list earlier: <*>`
- Sample9: `Host added was in lost list earlier: <*>`

Nearest labeled references:
- Index34, FAIL=False: `Got told to re-register updating block broadcast_14_piece0`; GT=`Got told to re-register updating block <*>`
- Index416, FAIL=False: `Registered BlockManager`; GT=`Registered BlockManager`
- Index346, FAIL=False: `Found block rdd_24186_8 locally`; GT=`Found block <*> locally`

All ten abstract host name correctly on unseen template, with a successful re-registration reference. Near-threshold TN is not strong evidence for a specific nearest-neighbor mechanism.


## C109: gemma/disjoint/Mac/TN

ID `mac-76316`, index443, seed0, slotnear_threshold; seen GT=False. Risk816.330174, threshold820.337930, percentile59.00%, margin/IQR-0.0059. First/last votes4/5,4/5; failure label=0. Other method percentiles: Clotho=59.0%, U_published=79.0%, lexical_LR=77.4%, raw_hidden_LR=39.4%, same_space_LR=46.2%, same_space_LR_real_only=33.0%, same_space_two_density=63.4%, structural_LR=51.4%

Raw log:
```text
DB Error: 1 "no such table: tb_c2cMsg_2171946401"
```
GT:
```text
DB Error: <*> "no such table: <*>"
```
Sample0:
```text
DB Error: <*> "no such table: <*>"
```

Target-batch parsed outputs:
- Sample5: `DB Error: <*> "no such table: <*>"`
- Sample6: `DB Error: <*> "no such table: <*>"`
- Sample7: `DB Error: <*> "no such table: <*>_<*>"`
- Sample8: `DB Error: <*> "no such table: <*>"`
- Sample9: `DB Error: <*> "no such table: <*>"`

Nearest labeled references:
- Index107, FAIL=True: `ARPT: 770595.414824: framerdy 0x0 bmccmd 3 framecnt 1024`; GT=`ARPT: <*>: framerdy <*> bmccmd <*> framecnt <*>`
- Index69, FAIL=True: `ARPT: 729787.205086: wl0: leaveModulePoweredForOffloads: Wi-Fi will turn off.`; GT=`ARPT: <*>: <*>: leaveModulePoweredForOffloads: Wi-Fi will turn off.`
- Index25, FAIL=True: `RTC: Maintenance 2017/7/3 06:08:08, sleep 2017/7/3 05:57:59`; GT=`RTC: Maintenance <*>, sleep <*>`

Database numeric/table fields mostly correct despite a failing hardware-status neighbor. Correct low risk can occur without semantically matching reference family.


## C110: gemma/disjoint/OpenStack/TN

ID `openstack-173848`, index307, seed0, slotseeded_representative; seen GT=False. Risk42.453406, threshold833.057909, percentile21.26%, margin/IQR-1.6957. First/last votes4/5,4/5; failure label=0. Other method percentiles: Clotho=21.3%, U_published=16.0%, lexical_LR=8.7%, raw_hidden_LR=5.1%, same_space_LR=5.1%, same_space_LR_real_only=7.1%, same_space_two_density=17.6%, structural_LR=71.1%

Raw log:
```text
[instance: 5cfb123d-30d6-49a8-9117-1838c899593c] Took 19.03 seconds to spawn the instance on the hypervisor.
```
GT:
```text
[instance: <*>] Took <*> seconds to spawn the instance on the hypervisor.
```
Sample0:
```text
[instance: <*>] Took <*> seconds to spawn the instance on the hypervisor.
```

Target-batch parsed outputs:
- Sample5: `[instance: <*>] Took 19.03 seconds to spawn the instance on the hypervisor.`
- Sample6: `[instance: <*>] Took <*> seconds to spawn the instance on the hypervisor.`
- Sample7: `[instance: <*>] Took <*> seconds to spawn the instance on the hypervisor.`
- Sample8: `[instance: <*>] Took <*> seconds to spawn the instance on the hypervisor.`
- Sample9: `[instance: <*>] Took <*> seconds to spawn the instance on the hypervisor.`

Nearest labeled references:
- Index237, FAIL=False: `[instance: 21c1242e-d320-42a7-9029-84d90a0c875c] Took 0.49 seconds to deallocate network for instance.`; GT=`[instance: <*>] Took <*> seconds to deallocate network for instance.`
- Index276, FAIL=False: `[instance: 6068079b-0861-48aa-84ac-8ddcef01ba97] Took 19.89 seconds to build instance.`; GT=`[instance: <*>] Took <*> seconds to build instance.`
- Index465, FAIL=False: `[instance: 6feeaa8f-b150-4be8-ba8c-eb74deac75c2] Took 20.29 seconds to build instance.`; GT=`[instance: <*>] Took <*> seconds to build instance.`

Instance timing sentence mostly succeeds near a successful deallocation-timing family. Useful cross-template generalization counters a pure identity-memorization account.


## C111: gemma/disjoint/Linux/TN

ID `linux-279`, index463, seed0, slotseeded_representative; seen GT=False. Risk174.084704, threshold602.287953, percentile10.20%, margin/IQR-1.0945. First/last votes5/5,5/5; failure label=0. Other method percentiles: Clotho=10.2%, U_published=16.2%, lexical_LR=1.8%, raw_hidden_LR=4.2%, same_space_LR=17.0%, same_space_LR_real_only=29.0%, same_space_two_density=8.2%, structural_LR=35.2%

Raw log:
```text
removing device node '/udev/vcsa2'
```
GT:
```text
removing device node <*>
```
Sample0:
```text
removing device node <*>
```

Target-batch parsed outputs:
- Sample5: `removing device node <*>`
- Sample6: `removing device node <*>`
- Sample7: `removing device node <*>`
- Sample8: `removing device node <*>`
- Sample9: `removing device node <*>`

Nearest labeled references:
- Index411, FAIL=False: `creating device node '/udev/vcs5'`; GT=`creating device node <*>`
- Index52, FAIL=False: `kjournald starting. Commit interval 5 seconds`; GT=`kjournald starting. Commit interval <*> seconds`
- Index344, FAIL=False: `inserting floppy driver for 2.6.5-1.358`; GT=`inserting floppy driver for <*>`

All ten mask device path correctly near a successful creating-device phrase. Structural/semantic family transfer succeeds despite disjoint GT templates.


## C112: gemma/disjoint/Mac/TN

ID `mac-52211`, index320, seed0, slotseeded_representative; seen GT=False. Risk307.446261, threshold820.337930, percentile10.20%, margin/IQR-0.7516. First/last votes0/5,3/5; failure label=0. Other method percentiles: Clotho=10.2%, U_published=43.0%, lexical_LR=52.6%, raw_hidden_LR=35.0%, same_space_LR=61.4%, same_space_LR_real_only=80.6%, same_space_two_density=17.4%, structural_LR=31.0%

Raw log:
```text
PM response took 108 ms (14437, LookupViewServic)
```
GT:
```text
PM response took <*> ms (<*>, <*>)
```
Sample0:
```text
PM response took <*> ms ( <*>, <*>)
```

Target-batch parsed outputs:
- Sample5: `PM response took <*> ms (<*, <*>)`
- Sample6: `PM response took <*> ms ( <*>, <*>)`
- Sample7: `PM response took <*> ms (<*>, <*>)`
- Sample8: `PM response took <*> ms (<*>, <*>)`
- Sample9: `PM response took <*> ms (<*>, <*>)`

Nearest labeled references:
- Index401, FAIL=False: `AppleThunderboltGenericHAL::earlyWake - complete - took 0 milliseconds`; GT=`AppleThunderboltGenericHAL::earlyWake - complete - took <*> milliseconds`
- Index344, FAIL=True: `WARNING: Type1 font data isn't in the correct format required by the Adobe Type 1 Font Format specification.`; GT=`WARNING: <*> font data isn't in the correct format required by the Adobe Type <*> Font Format specification.`
- Index107, FAIL=True: `ARPT: 770595.414824: framerdy 0x0 bmccmd 3 framecnt 1024`; GT=`ARPT: <*>: framerdy <*> bmccmd <*> framecnt <*>`

Timing/function tuple flips from 0/5 to 3/5 successes through whitespace and slot syntax. Low-risk TN is unstable; compare stable numeric TN C111 rather than claiming robust success.
