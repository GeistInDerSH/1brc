# One Billion Row Challenge

Link to original challenge: [Link](https://github.com/gunnarmorling/1brc)

## Results

Hardware:

* CPU:  AMD Ryzen 7 5800X 8-Core Processor
* RAM:  31Gi
* Disk: Samsung SSD 850 465Gi

### Default

```bash
# Cold Cache
\time -f 'Elapsed: %E' ./target/release/rs-1brc > /dev/null
Elapsed: 0:26.18

# Warm Cache
\time -f 'Elapsed: %E' ./target/release/rs-1brc > /dev/null
Elapsed: 0:01.29

# Use Profiling Guided Optimization
# i.e. run ./profile.sh first
\time -f 'Elapsed: %E' ./target/x86_64-unknown-linux-gnu/release/rs-1brc > /dev/null
Elapsed: 0:01.30
```

### 10K Key Set

```bash
# Cold Cache
\time -f 'Elapsed: %E' ./target/release/rs-1brc > /dev/null
Elapsed: 0:22.65

# Warm Cache
\time -f 'Elapsed: %E' ./target/release/rs-1brc > /dev/null
Elapsed: 0:01.70

# Use Profiling Guided Optimization
# i.e. run ./profile.sh first
\time -f 'Elapsed: %E' ./target/x86_64-unknown-linux-gnu/release/rs-1brc > /dev/null
Elapsed: 0:01.61
```

## Note

The [x64-no-dependencies](https://github.com/GeistInDerSH/1brc/tree/x64-no-dependencies) branch contains a version of
the code that can be compiled solely with `rustc`. This code however makes x86_64 Linux syscalls using assembly, and
should only work on those machines.