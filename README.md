# One Billion Row Challenge

Link to original challenge: [Link](https://github.com/gunnarmorling/1brc)

## Results

Hardware:

* CPU:  AMD Ryzen 7 5800X 8-Core Processor
* RAM:  31Gi
* Disk: Samsung SSD 850 465Gi

```bash
# Cold Cache
\time -f 'Elapsed: %E' ./target/release/rs-1brc > /dev/null
Elapsed: 0:26.16

# Warm Cache
\time -f 'Elapsed: %E' ./target/release/rs-1brc > /dev/null
Elapsed: 0:01.22

# Use Profiling Guided Optimization
# i.e. run ./profile.sh first
\time -f 'Elapsed: %E' ./target/x86_64-unknown-linux-gnu/release/rs-1brc > /dev/null
Elapsed: 0:01.26
```