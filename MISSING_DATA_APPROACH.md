# Missing Data Handling Approach

As requested, missing data points were filled using the following hierarchy:

1. **Missing Metro Values**: Used corresponding state value when available
2. **Missing State Values**: Used average of all metro values in that state
3. **Missing National Values**: Used average of all state values

## Implementation Details

The implementation can be found in the `handle_missing_data()` function, which:

- Processes each column independently
- Determines appropriate data type (text vs. numeric) for each column
- For numeric columns, uses mean for aggregation
- For text columns, uses mode (most common value) for aggregation
- Special attention is given to ensure the United States row has complete data
