# Frontier AI Logical Implication Evaluation Prompts

This file contains structured prompts generated directly from our **Transitive Relation Logic Chain (TRLC)** test set. 
You can manually paste these prompts into frontier AI systems like **Gemini 1.5 Pro**, **GPT-4o**, or **Claude 3.5 Sonnet** 
to evaluate their transitive reasoning accuracy at different complexity depths.


## --- Chain Length N = 2 ---

### Prompt 1 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x16 -> x17
x0 -> x3
x3 -> x18
x7 -> x9
x5 -> x12
x0 -> x7
x17 -> x3

Query:
Does x16 imply x3 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 2 (Ground Truth: **FALSE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x18 -> x3
x8 -> x10
x2 -> x19
x3 -> x9
x16 -> x2
x5 -> x3
x11 -> x2

Query:
Does x11 imply x10 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 3 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x6 -> x4
x7 -> x1
x1 -> x13
x11 -> x15
x4 -> x8
x1 -> x10
x14 -> x18

Query:
Does x6 imply x8 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 4 (Ground Truth: **FALSE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x18 -> x19
x9 -> x11
x18 -> x6
x1 -> x8
x19 -> x0
x12 -> x3
x7 -> x0

Query:
Does x19 imply x8 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 5 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x2 -> x8
x2 -> x0
x1 -> x19
x19 -> x9
x19 -> x15
x3 -> x4
x5 -> x0

Query:
Does x1 imply x15 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```


## --- Chain Length N = 4 ---

### Prompt 1 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x0 -> x2
x1 -> x18
x17 -> x1
x7 -> x16
x12 -> x1
x18 -> x7
x10 -> x1
x4 -> x18
x0 -> x4

Query:
Does x12 imply x16 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 2 (Ground Truth: **FALSE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x4 -> x13
x14 -> x2
x7 -> x10
x11 -> x2
x15 -> x14
x2 -> x17
x8 -> x4
x3 -> x19
x16 -> x14

Query:
Does x4 imply x17 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 3 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x17 -> x5
x3 -> x11
x2 -> x9
x11 -> x16
x7 -> x19
x3 -> x18
x5 -> x17
x18 -> x7
x19 -> x10

Query:
Does x3 imply x10 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 4 (Ground Truth: **FALSE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x7 -> x11
x12 -> x18
x9 -> x6
x12 -> x14
x12 -> x17
x0 -> x16
x8 -> x15
x18 -> x0
x5 -> x4

Query:
Does x9 imply x16 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 5 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x12 -> x10
x1 -> x12
x6 -> x12
x16 -> x10
x1 -> x6
x3 -> x1
x7 -> x8
x12 -> x15
x8 -> x19

Query:
Does x3 imply x10 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```


## --- Chain Length N = 6 ---

### Prompt 1 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x6 -> x13
x9 -> x16
x15 -> x7
x13 -> x5
x4 -> x15
x6 -> x16
x17 -> x16
x4 -> x3
x17 -> x9
x4 -> x17
x16 -> x6

Query:
Does x4 imply x5 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 2 (Ground Truth: **FALSE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x18 -> x4
x7 -> x8
x8 -> x0
x9 -> x18
x12 -> x19
x10 -> x3
x11 -> x13
x3 -> x7
x5 -> x6
x11 -> x3
x0 -> x18

Query:
Does x10 imply x9 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 3 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x13 -> x3
x18 -> x12
x10 -> x18
x19 -> x8
x4 -> x11
x15 -> x13
x3 -> x0
x11 -> x10
x0 -> x11
x6 -> x14
x18 -> x7

Query:
Does x13 imply x12 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 4 (Ground Truth: **FALSE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x19 -> x14
x18 -> x2
x3 -> x10
x1 -> x3
x16 -> x14
x0 -> x14
x16 -> x2
x3 -> x14
x15 -> x5
x14 -> x16
x9 -> x1

Query:
Does x0 imply x10 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 5 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x6 -> x18
x3 -> x4
x12 -> x13
x7 -> x14
x7 -> x8
x10 -> x12
x6 -> x4
x1 -> x2
x13 -> x2
x2 -> x7
x14 -> x3

Query:
Does x10 imply x3 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```


## --- Chain Length N = 8 ---

### Prompt 1 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x2 -> x19
x11 -> x7
x10 -> x2
x11 -> x10
x4 -> x3
x7 -> x6
x7 -> x15
x15 -> x10
x8 -> x11
x17 -> x4
x2 -> x10
x13 -> x2
x2 -> x17

Query:
Does x8 imply x3 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 2 (Ground Truth: **FALSE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x14 -> x15
x3 -> x13
x15 -> x18
x17 -> x3
x8 -> x19
x16 -> x9
x19 -> x6
x17 -> x4
x6 -> x17
x18 -> x14
x4 -> x8
x13 -> x16
x2 -> x11

Query:
Does x18 imply x13 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 3 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x7 -> x8
x12 -> x7
x4 -> x8
x8 -> x12
x1 -> x2
x13 -> x12
x16 -> x13
x18 -> x3
x7 -> x10
x14 -> x18
x10 -> x1
x3 -> x4
x1 -> x16

Query:
Does x18 imply x16 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 4 (Ground Truth: **FALSE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x11 -> x0
x5 -> x13
x9 -> x5
x0 -> x9
x2 -> x9
x5 -> x4
x11 -> x17
x1 -> x4
x15 -> x2
x16 -> x3
x10 -> x11
x2 -> x0
x17 -> x16

Query:
Does x10 imply x13 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 5 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x12 -> x1
x17 -> x3
x9 -> x10
x8 -> x16
x4 -> x3
x1 -> x4
x19 -> x9
x15 -> x9
x4 -> x17
x10 -> x12
x3 -> x17
x0 -> x16
x16 -> x19

Query:
Does x0 imply x17 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```


## --- Chain Length N = 10 ---

### Prompt 1 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x14 -> x0
x1 -> x16
x0 -> x17
x7 -> x2
x13 -> x4
x11 -> x13
x15 -> x7
x15 -> x11
x4 -> x5
x8 -> x15
x15 -> x13
x16 -> x15
x5 -> x14
x17 -> x12
x3 -> x1

Query:
Does x1 imply x12 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 2 (Ground Truth: **FALSE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x16 -> x15
x5 -> x13
x4 -> x19
x4 -> x3
x6 -> x16
x13 -> x18
x0 -> x6
x15 -> x14
x6 -> x17
x14 -> x4
x2 -> x17
x19 -> x0
x18 -> x13
x6 -> x9
x18 -> x2

Query:
Does x5 imply x3 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 3 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x3 -> x9
x12 -> x16
x2 -> x12
x8 -> x2
x4 -> x2
x14 -> x1
x16 -> x3
x7 -> x10
x5 -> x17
x4 -> x1
x6 -> x11
x1 -> x19
x19 -> x0
x0 -> x11
x11 -> x8

Query:
Does x4 imply x9 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 4 (Ground Truth: **FALSE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x15 -> x7
x15 -> x8
x10 -> x9
x18 -> x19
x3 -> x16
x12 -> x8
x7 -> x17
x15 -> x0
x0 -> x5
x9 -> x11
x2 -> x15
x5 -> x12
x8 -> x2
x1 -> x3
x1 -> x10

Query:
Does x0 imply x11 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```

### Prompt 5 (Ground Truth: **TRUE**)

```text
You are a rigorous logical reasoning system. Evaluate the following set of implication rules and determine if the query implies a connection.

Rules:
x4 -> x1
x16 -> x9
x10 -> x14
x13 -> x2
x2 -> x16
x1 -> x2
x3 -> x10
x6 -> x3
x17 -> x0
x11 -> x18
x11 -> x4
x7 -> x11
x14 -> x7
x16 -> x8
x14 -> x4

Query:
Does x6 imply x8 through a chain of rules?

Think step-by-step to verify the chain of implication. Once you have traced the sequence, output your final answer on a new line EXACTLY as either 'Answer: TRUE' or 'Answer: FALSE'.
```
