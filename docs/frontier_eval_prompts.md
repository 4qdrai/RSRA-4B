# Frontier AI Logical Implication Evaluation Prompts

This file contains structured prompts generated directly from our **Transitive Relation Logic Chain (TRLC)** test set. 
You can manually paste these prompts into frontier AI systems like **Gemini 1.5 Pro**, **GPT-4o**, or **Claude 3.5 Sonnet** 
to evaluate their transitive reasoning accuracy at different complexity depths.


## --- Chain Length N = 2 ---

### Prompt 1 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x33 -> x34
x0 -> x6
x7 -> x36
x14 -> x18
x11 -> x24
x1 -> x14
x34 -> x7

Query:
Does x33 imply x7 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 2 (Ground Truth: **FALSE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x37 -> x6
x17 -> x20
x4 -> x39
x6 -> x18
x32 -> x4
x10 -> x7
x22 -> x4

Query:
Does x22 imply x21 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 3 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x12 -> x8
x15 -> x3
x2 -> x27
x23 -> x30
x8 -> x16
x2 -> x21
x29 -> x37

Query:
Does x12 imply x16 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 4 (Ground Truth: **FALSE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x2 -> x17
x36 -> x12
x18 -> x22
x35 -> x34
x14 -> x0
x39 -> x0
x37 -> x38

Query:
Does x39 imply x17 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 5 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x18 -> x4
x17 -> x4
x0 -> x39
x3 -> x39
x30 -> x7
x9 -> x11
x27 -> x3

Query:
Does x27 imply x39 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```


## --- Chain Length N = 4 ---

### Prompt 1 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x1 -> x4
x2 -> x37
x34 -> x2
x14 -> x33
x25 -> x2
x37 -> x14
x20 -> x3
x9 -> x37
x1 -> x8

Query:
Does x25 imply x33 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 2 (Ground Truth: **FALSE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x9 -> x27
x29 -> x4
x14 -> x21
x22 -> x4
x30 -> x29
x4 -> x34
x17 -> x9
x6 -> x38
x32 -> x28

Query:
Does x9 imply x34 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 3 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x37 -> x15
x7 -> x23
x4 -> x18
x36 -> x37
x35 -> x11
x23 -> x33
x7 -> x37
x39 -> x20
x15 -> x39

Query:
Does x7 imply x20 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 4 (Ground Truth: **FALSE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x19 -> x12
x24 -> x34
x36 -> x1
x25 -> x28
x25 -> x36
x5 -> x4
x1 -> x33
x11 -> x8
x15 -> x23

Query:
Does x19 imply x33 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 5 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x24 -> x2
x8 -> x24
x24 -> x20
x32 -> x20
x2 -> x6
x17 -> x39
x3 -> x25
x12 -> x7
x6 -> x3

Query:
Does x8 imply x3 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```


## --- Chain Length N = 6 ---

### Prompt 1 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x32 -> x12
x12 -> x27
x34 -> x19
x10 -> x9
x32 -> x9
x14 -> x13
x32 -> x35
x35 -> x34
x8 -> x35
x19 -> x32
x6 -> x31

Query:
Does x8 imply x27 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 2 (Ground Truth: **FALSE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x27 -> x11
x38 -> x17
x12 -> x25
x21 -> x6
x36 -> x18
x18 -> x0
x8 -> x22
x6 -> x15
x35 -> x18
x15 -> x17
x3 -> x35

Query:
Does x3 imply x17 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 3 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x26 -> x20
x14 -> x9
x20 -> x27
x30 -> x34
x9 -> x21
x23 -> x27
x25 -> x30
x21 -> x30
x34 -> x20
x0 -> x37
x7 -> x0

Query:
Does x14 imply x27 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 4 (Ground Truth: **FALSE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x25 -> x36
x9 -> x10
x10 -> x17
x17 -> x37
x37 -> x35
x27 -> x0
x18 -> x2
x29 -> x33
x27 -> x36
x6 -> x1
x14 -> x9

Query:
Does x14 imply x36 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 5 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x10 -> x13
x5 -> x30
x15 -> x22
x37 -> x5
x2 -> x17
x30 -> x10
x2 -> x25
x38 -> x12
x13 -> x4
x28 -> x37
x34 -> x27

Query:
Does x28 imply x4 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```


## --- Chain Length N = 8 ---

### Prompt 1 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x15 -> x30
x35 -> x5
x21 -> x5
x17 -> x23
x30 -> x21
x20 -> x14
x12 -> x4
x5 -> x34
x21 -> x38
x23 -> x15
x6 -> x22
x34 -> x20
x9 -> x22

Query:
Does x17 imply x14 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 2 (Ground Truth: **FALSE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x37 -> x29
x28 -> x34
x29 -> x8
x6 -> x12
x33 -> x34
x31 -> x27
x27 -> x6
x17 -> x37
x37 -> x7
x26 -> x39
x8 -> x39
x12 -> x28
x7 -> x34

Query:
Does x27 imply x39 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 3 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x30 -> x34
x14 -> x10
x26 -> x30
x10 -> x36
x2 -> x33
x37 -> x26
x34 -> x14
x15 -> x16
x3 -> x5
x15 -> x21
x6 -> x8
x36 -> x6
x16 -> x24

Query:
Does x37 imply x8 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 4 (Ground Truth: **FALSE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x22 -> x34
x27 -> x12
x0 -> x19
x34 -> x5
x28 -> x36
x19 -> x10
x7 -> x5
x8 -> x3
x0 -> x10
x20 -> x22
x19 -> x20
x4 -> x28
x36 -> x0

Query:
Does x4 imply x33 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 5 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x30 -> x18
x1 -> x33
x22 -> x34
x20 -> x2
x21 -> x24
x38 -> x18
x31 -> x25
x9 -> x34
x34 -> x9
x14 -> x1
x9 -> x14
x25 -> x27
x27 -> x22

Query:
Does x31 imply x33 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```


## --- Chain Length N = 10 ---

### Prompt 1 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x22 -> x28
x34 -> x24
x2 -> x8
x30 -> x23
x23 -> x26
x15 -> x4
x11 -> x22
x30 -> x14
x3 -> x32
x32 -> x30
x8 -> x10
x28 -> x0
x10 -> x11
x26 -> x2
x0 -> x2

Query:
Does x3 imply x28 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 2 (Ground Truth: **FALSE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x10 -> x7
x25 -> x13
x36 -> x38
x11 -> x5
x29 -> x8
x33 -> x30
x20 -> x27
x11 -> x17
x33 -> x11
x27 -> x36
x1 -> x12
x38 -> x1
x8 -> x38
x5 -> x20
x17 -> x5

Query:
Does x25 imply x12 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 3 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x33 -> x24
x8 -> x33
x39 -> x24
x12 -> x4
x4 -> x2
x0 -> x23
x9 -> x15
x25 -> x2
x2 -> x39
x24 -> x12
x15 -> x33
x38 -> x9
x16 -> x4
x9 -> x6
x39 -> x0

Query:
Does x38 imply x23 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 4 (Ground Truth: **FALSE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x13 -> x23
x35 -> x18
x19 -> x16
x23 -> x16
x16 -> x12
x20 -> x11
x12 -> x8
x20 -> x10
x3 -> x14
x14 -> x35
x5 -> x30
x25 -> x16
x5 -> x3
x14 -> x3
x8 -> x5

Query:
Does x20 imply x18 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 5 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x3 -> x18
x6 -> x14
x1 -> x20
x21 -> x29
x38 -> x3
x4 -> x16
x33 -> x10
x13 -> x4
x13 -> x7
x18 -> x33
x30 -> x28
x7 -> x30
x28 -> x13
x16 -> x38
x32 -> x15

Query:
Does x7 imply x10 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```
