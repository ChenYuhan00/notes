# AI Chips

## Out-of-Order execution

顺序Pipeline经常被install，Load从内存里读东西可能超过200个指令周期

Move the dependent instructions out of the way of indeoendent ones

```text
MUL  R3 <- R1,R2 F|D|E|E|E|E|R|W
ADD  R3 <- R3,R1   F|D|-|-|-|E|R|W
ADD  R1 <- R6,R7     F|-|-|-|D|E|R|W
MUL  R5 <- R6,R8             F|D|E|E|E|E|R|W
ADD  R7 <- R3,R5               F|D|-|-|-|E|R|W
16 cycles
```

```text
MUL  R3 <- R1,R2 F|D|E|E|E|E|R|W
ADD  R3 <- R3,R1   F|D|wait |E|R|W
ADD  R1 <- R6,R7     F|D|E|R|-|-|-|W
MUL  R5 <- R6,R8       F|D|E|E|E|E|R|W
ADD  R7 <- R3,R5         F|D|wait |E|R|W
(更改顺序后)12 cycles
```

### Tomasulo’s Algorithms

Hump1:Reservation stations
Hump2: Reorder buffer

#### Enabling OoO Execution

1. Need to link the consumer of a value to the producer
2. Need to buffer instructions until they are read to execute
3. instruction need to keep track of radiness of source values
4. When all source values of an instruction are ready,need to dispatch the instruction to its functional unit

...

不如看[Q](https://note.hobbitqia.cc/CA/CA4/#tomasulos-approach)

## 复习

- Amdhal Law
- Roofline Model
- Little's Law

`PPT9 10:20`

- Reorder Buffer
- Dispatch stall
- Tomasulo's algorithm

`PPT61 10:45`

- Performace Analysis
- Flynn's Taxonomy of Computers
- Hardware execution model

- Memory
- FF SRAM DRAM
- DRAM Access States refresh
- cache Memory Hierarchy

`PPT111 11:35`

- Chache Coherence
- MSI protocol

- AI Acceletator
- 图优化
