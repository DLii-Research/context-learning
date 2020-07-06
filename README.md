# n-task Expremintation

This repository contains various experiments using n-task layers and models

# Implementation Details:

## Dynamic n-task

When working with n-task, context switching is determined by achieving a large enough loss delta that exceeds a threshold. This is calculated using the expected loss vs. the actual loss (`delta = atr_value - context_loss`). If this value exceeds a threshold (less than zero), then a task switch is triggered.

Similarly, dynamic n-task also looks at this delta. If the delta exceeds the task switch case in all contexts, and the best fitting context (`max(atr_value - context_loss)`) is less than the specified threshold, a new task is added.